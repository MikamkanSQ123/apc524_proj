from pytest import approx
from pathlib import Path
from simple_backtester.backtest import Backtester


def test_backtest_evaluate_1():
    # Load the YAML configuration and the new strategy
    strategy_module_path = "tests/test_data/strategy/strat1.py"
    config_path = "tests/test_data/strategy/strat1.yaml"
    # config = yaml.safe_load(Path(config_path).read_text())

    # Dynamically load the strategy class
    backtest = Backtester(strategy_module_path, config_path)
    source = "local"
    path = f"tests/test_data/data/{source}/feature/"
    config = {
        "source": source,
        "data_path": path,
        # "tech_indicators": ["ma", "macd", "rsi"],
        "features": [file.name[:-4] for file in Path(path).iterdir() if file.is_file()],
    }
    backtest.run(config)

    # Get and display results
    results = backtest.get_results()
    assert results["cumulative_pnl"] == approx(0.013461453243535369)
    assert results["Sharpe Ratio"] == approx(0.01892717180245496)
    assert results["Volatility"] == approx(0.0006567162552354367)
    assert results["Max Drawdown"] == approx(0.01724343880681582)
    assert results["pnl_history"][:10] == approx(
        [
            -2.37813288e-04,
            -2.46216033e-19,
            -4.66596038e-19,
            -2.46579239e-19,
            -0.00000000e00,
            -2.62368353e-19,
            1.90459212e-19,
            3.17184983e-03,
            -0.00000000e00,
            -0.00000000e00,
        ]
    )


def test_backtest_evaluate_2():
    # Load the YAML configuration and the new strategy
    strategy_module_path = "tests/test_data/strategy/strat0.py"
    config_path = "tests/test_data/strategy/strat0.yaml"
    # config = yaml.safe_load(Path(config_path).read_text())

    # Dynamically load the strategy class
    backtest = Backtester(strategy_module_path, config_path)
    source = "local"
    path = f"tests/test_data/data/{source}/feature/"
    config = {
        "source": source,
        "data_path": path,
        # "tech_indicators": ["ma", "macd", "rsi"],
        "features": [file.name[:-4] for file in Path(path).iterdir() if file.is_file()],
    }
    backtest.run(config)

    # Get and display results
    results = backtest.get_results()
    assert results["cumulative_pnl"] == approx(0.0)
    assert results["Sharpe Ratio"] == approx(0.0)
    assert results["Volatility"] == approx(0.0)
    assert results["Max Drawdown"] == approx(0.0)
    assert results["pnl_history"][:10] == approx(0.0)
