import pytest
from simple_backtester import Strategy
from simple_backtester.config import RiskConfig
import dataclasses


def test_strategy_abstract():
    with pytest.raises(TypeError):
        Strategy()


def test_strategy_inheritance():
    class TestStrategy(Strategy):
        def evaluate(self):
            pass

    tstrat = TestStrategy("tests/test_data/strategy/strat1.yaml")
    assert isinstance(tstrat, Strategy)
    assert isinstance(tstrat, TestStrategy)
    assert tstrat.setup.warm_up == 100
    assert tstrat.setup.look_back == 100
    assert tstrat.parameters.moving_average_window == 20
    assert tstrat.risk.stop_loss == 0.02

    with pytest.raises(dataclasses.FrozenInstanceError):
        tstrat.setup.look_back = 200

    with pytest.raises(AttributeError):
        tstrat.parameters = None

    with pytest.raises(AttributeError):
        tstrat.risk = RiskConfig(stop_loss=0.01)

    tstrat.parameters.moving_average_window = 50
    assert tstrat.parameters.moving_average_window == 50
