import pytest
from pytest import approx
from simple_backtester import Strategy
from simple_backtester.config import RiskConfig
import dataclasses
import importlib.util
import numpy as np


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
    assert tstrat.parameters.ma_base == 10
    assert tstrat.risk.stop_loss == 0.02

    with pytest.raises(dataclasses.FrozenInstanceError):
        tstrat.setup.look_back = 200

    with pytest.raises(AttributeError):
        tstrat.parameters = None

    with pytest.raises(AttributeError):
        tstrat.risk = RiskConfig(stop_loss=0.01)

    tstrat.parameters.ma_base = 50
    assert tstrat.parameters.ma_base == 50


def test_strategy_evaluate():
    spec = importlib.util.spec_from_file_location(
        "MeanReversion", "tests/test_data/strategy/strat1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MeanReversion = module.MeanReversion
    tstrat = MeanReversion("tests/test_data/strategy/strat1.yaml")
    data = np.arange(0, 300).reshape(100, 3).cumsum(axis=0)
    weights = tstrat.evaluate(data)
    assert weights == approx(np.array([-1.22694033, 0.00440279, 1.22253754]))
