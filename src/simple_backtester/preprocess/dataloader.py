import pandas as pd
from typing import Any

config = {
    "data_path": "simple_backtester/data/feature/return.csv",
    "tech_indicators": ["moving_average"],
    "features": ["return"],
}


class DataLoader:
    def __init__(self, config: dict[str, Any] = config):
        self.config = config
        print(f'Loading data from {config["data_path"]}')
        self.data = pd.read_csv(config["data_path"], index_col=0, parse_dates=True)

    def load_data(
        self,
        start: str,
        end: str,
        symbols: list[str],
        features: list[str],
        args: dict[str, Any],
    ) -> dict[str, pd.DataFrame]:
        features_dict: dict[str, Any] = {}
        for feature in features:
            features_dict[feature] = None
            if feature in self.config["features"]:
                features_dict[feature] = self.data[symbols][start:end]  # type: ignore[misc]
            elif feature in self.config["tech_indicators"]:
                features_dict[feature] = eval(
                    f"self.{feature}(start, end, symbols, args.get(feature, None))"
                )
        return features_dict

    def moving_average(
        self, start: str, end: str, symbols: list[str], arg: int
    ) -> pd.DataFrame:
        return self.data[symbols].rolling(window=arg).mean()[start:end]  # type: ignore[misc]


if __name__ == "__main__":
    dl = DataLoader()
    features = dl.load_data(
        "2024-11-11 00:00:00",
        "2024-11-11 23:54:00",
        ["BTC"],
        ["moving_average", "return"],
        {"moving_average": 10},
    )
