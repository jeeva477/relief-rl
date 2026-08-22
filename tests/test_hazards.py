from rl.envs.hazard import Hazard, HazardType


def test_hazard_contains_point_inside_radius():
    h = Hazard(id="h1", x=0.5, y=0.5, radius=0.1, severity=0.5)
    assert h.contains(0.5, 0.5)
    assert h.contains(0.55, 0.5)
    assert not h.contains(0.8, 0.8)


def test_hazard_risk_decays_with_distance():
    h = Hazard(id="h1", x=0.5, y=0.5, radius=0.1, severity=0.8)
    risk_center = h.risk_at(0.5, 0.5)
    # 0.15 is outside the hard radius (0.1) but inside the falloff zone
    # (radius * falloff = 0.2 by default), so it should be partially decayed.
    risk_near = h.risk_at(0.65, 0.5)
    risk_far = h.risk_at(0.95, 0.95)
    assert risk_center == 0.8
    assert 0 < risk_near < risk_center
    assert risk_far == 0.0


def test_hazard_moves_with_velocity():
    h = Hazard(id="h1", x=0.5, y=0.5, radius=0.1, severity=0.5, velocity=(0.05, 0.0))
    h.step(dt=1.0)
    assert h.x == 0.55
    assert h.y == 0.5


def test_hazard_grows_with_growth_rate():
    h = Hazard(id="h1", x=0.5, y=0.5, radius=0.1, severity=0.5, growth_rate=0.02)
    h.step(dt=1.0)
    assert h.radius == pytest_approx(0.12)


def pytest_approx(v, tol=1e-6):
    class _Approx(float):
        def __eq__(self, other):
            return abs(other - v) < tol
    return _Approx(v)


def test_inactive_hazard_contains_nothing():
    h = Hazard(id="h1", x=0.5, y=0.5, radius=0.5, severity=1.0, active=False)
    assert not h.contains(0.5, 0.5)
    assert h.risk_at(0.5, 0.5) == 0.0


def test_hard_constraint_flag_independent_of_severity():
    h = Hazard(id="h1", x=0.5, y=0.5, radius=0.1, severity=0.1, hazard_type=HazardType.CHEMICAL_LEAK,
               hard_constraint=True)
    assert h.hard_constraint
    assert h.severity == 0.1  # low soft severity but still a hard exclusion zone
