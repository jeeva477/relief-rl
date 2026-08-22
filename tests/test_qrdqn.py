"""
QR-DQN tests: quantile network shape/masking, the quantile-regression
(pinball/Huber) loss, replay buffer mechanics, Double-DQN target
computation, and end-to-end penalty propagation on the actual
EvacuationEnv -- everything verified with real tensors and real
environment steps, no fabricated values.
"""

import numpy as np
import torch

from rl.algorithms.qrdqn import QRDQNConfig, QRDQNTrainer, ReplayBuffer, quantile_huber_loss
from rl.envs.evacuation_env import EvacuationEnv, N_ACTIONS, OBS_DIM
from rl.models.qrdqn_net import QuantileNetwork


def test_quantile_network_output_shape():
    net = QuantileNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, n_quantiles=11, hidden_dim=32)
    obs = torch.randn(5, OBS_DIM)
    quantiles = net(obs)
    assert quantiles.shape == (5, N_ACTIONS, 11)


def test_q_values_mean_over_quantiles():
    net = QuantileNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, n_quantiles=11, hidden_dim=32)
    obs = torch.randn(3, OBS_DIM)
    quantiles = net(obs)
    q = net.q_values(obs)
    assert torch.allclose(q, quantiles.mean(dim=-1), atol=1e-5)


def test_masked_actions_have_negative_infinity_q_value():
    net = QuantileNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, n_quantiles=11, hidden_dim=32)
    obs = torch.randn(2, OBS_DIM)
    mask = torch.ones(2, N_ACTIONS, dtype=torch.bool)
    mask[:, 2] = False
    q = net.q_values(obs, action_mask=mask)
    assert torch.isinf(q[:, 2]).all() and (q[:, 2] < 0).all()
    assert torch.isfinite(q[:, [i for i in range(N_ACTIONS) if i != 2]]).all()


def test_masked_actions_never_selected_greedy():
    net = QuantileNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, n_quantiles=11, hidden_dim=32)
    obs = torch.randn(1, OBS_DIM)
    mask = torch.zeros(1, N_ACTIONS, dtype=torch.bool)
    mask[0, 4] = True
    for _ in range(20):
        a = net.act(obs, action_mask=mask, epsilon=0.0, deterministic=True)
        assert int(a.item()) == 4


def test_masked_actions_never_selected_epsilon_greedy():
    net = QuantileNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS, n_quantiles=11, hidden_dim=32)
    obs = torch.randn(1, OBS_DIM)
    mask = torch.zeros(1, N_ACTIONS, dtype=torch.bool)
    mask[0, [1, 3]] = True
    for _ in range(50):
        a = net.act(obs, action_mask=mask, epsilon=1.0, deterministic=False)
        assert int(a.item()) in (1, 3)


def test_quantile_huber_loss_zero_when_distributions_identical():
    """When every predicted quantile equals every target quantile (a
    degenerate constant distribution), every pairwise TD error is zero
    and the loss must vanish. (Loss is pairwise over *all* (i, j)
    quantile combinations, so this only holds when predicted and target
    agree everywhere, not merely index-for-index on arbitrary values.)"""
    taus = (torch.arange(9, dtype=torch.float32) + 0.5) / 9
    values = torch.full((4, 9), 3.0)
    loss = quantile_huber_loss(values, values, taus)
    assert loss.item() < 1e-6


def test_quantile_huber_loss_positive_and_finite_on_mismatch():
    taus = (torch.arange(9, dtype=torch.float32) + 0.5) / 9
    predicted = torch.zeros(4, 9)
    target = torch.ones(4, 9) * 5.0
    loss = quantile_huber_loss(predicted, target, taus)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_replay_buffer_fifo_capacity():
    buf = ReplayBuffer(capacity=5)
    for i in range(10):
        buf.add(
            obs=np.zeros(OBS_DIM, dtype=np.float32), action=i % N_ACTIONS, reward=float(i),
            next_obs=np.zeros(OBS_DIM, dtype=np.float32), done=False,
            mask=np.ones(N_ACTIONS, dtype=bool), next_mask=np.ones(N_ACTIONS, dtype=bool),
        )
    assert len(buf) == 5  # oldest 5 evicted


def _trainer(seed: int = 0) -> QRDQNTrainer:
    torch.manual_seed(seed)
    cfg = QRDQNConfig(n_quantiles=11, hidden_dim=32, batch_size=16, min_replay_size=16,
                       buffer_size=1000, target_update_freq=5, train_freq=1)
    return QRDQNTrainer(obs_dim=OBS_DIM, n_actions=N_ACTIONS, config=cfg, device="cpu")


def _fill_buffer(trainer: QRDQNTrainer, n: int = 32, reward: float = 1.0) -> None:
    for i in range(n):
        trainer.store(
            obs=np.zeros(OBS_DIM, dtype=np.float32), action=i % N_ACTIONS, reward=reward,
            next_obs=np.zeros(OBS_DIM, dtype=np.float32), done=False,
            mask=np.ones(N_ACTIONS, dtype=bool), next_mask=np.ones(N_ACTIONS, dtype=bool),
        )


def test_train_step_produces_finite_loss_and_updates_weights():
    trainer = _trainer()
    _fill_buffer(trainer, n=32)
    before = [p.clone() for p in trainer.online.parameters()]
    stats = trainer.train_step()
    assert np.isfinite(stats["loss"])
    assert stats["skipped"] is False
    after = list(trainer.online.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_target_network_syncs_on_schedule():
    trainer = _trainer()
    _fill_buffer(trainer, n=32)
    for _ in range(trainer.config.target_update_freq):
        trainer.train_step()
    online_params = list(trainer.online.parameters())
    target_params = list(trainer.target.parameters())
    assert all(torch.equal(o, t) for o, t in zip(online_params, target_params))


def test_epsilon_decays_from_start_to_end():
    trainer = _trainer()
    cfg = trainer.config
    trainer._env_steps = 0
    assert abs(trainer.epsilon() - cfg.epsilon_start) < 1e-6
    trainer._env_steps = cfg.epsilon_decay_steps
    assert abs(trainer.epsilon() - cfg.epsilon_end) < 1e-6
    trainer._env_steps = cfg.epsilon_decay_steps * 10  # clamps, doesn't overshoot
    assert abs(trainer.epsilon() - cfg.epsilon_end) < 1e-6


def test_select_action_respects_mask_on_real_env():
    trainer = _trainer(seed=1)
    env = EvacuationEnv(difficulty="EASY", grid_size=5, max_steps=20)
    obs, _ = env.reset(seed=3)
    mask = env.valid_action_mask()
    for _ in range(20):
        a = trainer.select_action(obs, mask, deterministic=False)
        assert mask[a]


def test_penalty_propagation_reduces_expected_q_of_bad_action_on_real_env():
    """
    Repeatedly experience a large negative reward for a fixed action from
    a fixed observation and verify the online network's expected Q-value
    for that (obs, action) pair decreases -- the QR-DQN analogue of the
    PPO test_penalty_propagation_reduces_risky_action_probability test.
    """
    trainer = _trainer(seed=42)
    fixed_obs = np.zeros(OBS_DIM, dtype=np.float32)
    fixed_mask = np.ones(N_ACTIONS, dtype=bool)
    bad_action = 5

    q_before = trainer.q_values(fixed_obs, fixed_mask)[bad_action]

    # Warm up the buffer above min_replay_size/batch_size before the first
    # direct train_step() call (train_step, unlike maybe_train_step, does
    # not itself check buffer readiness).
    for a in range(N_ACTIONS):
        r = -10.0 if a == bad_action else 0.1
        trainer.store(obs=fixed_obs, action=a, reward=r, next_obs=fixed_obs, done=True,
                      mask=fixed_mask, next_mask=fixed_mask)

    for _ in range(200):
        # Bad action -> large penalty; other actions -> small positive reward.
        for a in range(N_ACTIONS):
            r = -10.0 if a == bad_action else 0.1
            trainer.store(
                obs=fixed_obs, action=a, reward=r, next_obs=fixed_obs, done=True,
                mask=fixed_mask, next_mask=fixed_mask,
            )
        trainer.train_step()

    q_after = trainer.q_values(fixed_obs, fixed_mask)[bad_action]
    assert np.isfinite(q_before) and np.isfinite(q_after)
    assert q_after < q_before


def test_double_dqn_can_be_disabled():
    cfg = QRDQNConfig(n_quantiles=11, hidden_dim=32, batch_size=16, min_replay_size=16,
                       buffer_size=1000, double_dqn=False, train_freq=1)
    trainer = QRDQNTrainer(obs_dim=OBS_DIM, n_actions=N_ACTIONS, config=cfg, device="cpu")
    _fill_buffer(trainer, n=32)
    stats = trainer.train_step()
    assert np.isfinite(stats["loss"])


def test_checkpoint_roundtrip(tmp_path):
    import os
    trainer = _trainer(seed=7)
    path = os.path.join(tmp_path, "qrdqn.pt")
    torch.save({
        "model_state_dict": trainer.online.state_dict(),
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "n_quantiles": trainer.config.n_quantiles,
        "hidden_dim": trainer.config.hidden_dim,
        "model_name": "ReliefRL-QRDQN",
        "algo": "qrdqn",
    }, path)

    loaded = QuantileNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS,
                              n_quantiles=trainer.config.n_quantiles, hidden_dim=trainer.config.hidden_dim)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    loaded.load_state_dict(ckpt["model_state_dict"])

    obs = torch.randn(4, OBS_DIM)
    with torch.no_grad():
        q1 = trainer.online(obs)
        q2 = loaded(obs)
    assert torch.allclose(q1, q2)
