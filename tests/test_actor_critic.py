import torch

from rl.models.actor_critic import ActorCritic


def test_forward_shapes():
    model = ActorCritic(obs_dim=27, n_actions=5, hidden_dim=32)
    obs = torch.randn(4, 27)
    logits, value = model(obs)
    assert logits.shape == (4, 5)
    assert value.shape == (4,)


def test_get_action_returns_valid_action():
    model = ActorCritic(obs_dim=27, n_actions=5, hidden_dim=32)
    obs = torch.randn(1, 27)
    action, log_prob, value = model.get_action(obs)
    assert 0 <= int(action.item()) < 5
    assert log_prob.shape == (1,)
    assert value.shape == (1,)


def test_action_mask_excludes_masked_actions():
    model = ActorCritic(obs_dim=27, n_actions=5, hidden_dim=32)
    obs = torch.randn(1, 27)
    mask = torch.tensor([[True, False, False, False, False]])
    for _ in range(20):
        action, _, _ = model.get_action(obs, action_mask=mask)
        assert int(action.item()) == 0


def test_deterministic_action_is_argmax():
    model = ActorCritic(obs_dim=27, n_actions=5, hidden_dim=32)
    obs = torch.randn(1, 27)
    logits, _ = model(obs)
    expected = int(torch.argmax(logits, dim=-1).item())
    action, _, _ = model.get_action(obs, deterministic=True)
    assert int(action.item()) == expected


def test_evaluate_actions_shapes():
    model = ActorCritic(obs_dim=27, n_actions=5, hidden_dim=32)
    obs = torch.randn(8, 27)
    actions = torch.randint(0, 5, (8,))
    log_probs, entropy, values = model.evaluate_actions(obs, actions)
    assert log_probs.shape == (8,)
    assert entropy.shape == (8,)
    assert values.shape == (8,)
