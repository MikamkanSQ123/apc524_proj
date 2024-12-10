import pandas as pd
from typing import Any, Union

config = {
    "data_path": "./src/simple_backtester/data/feature/price.csv",
    "tech_indicators": ["ma", "macd", "rsi", "return"],
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
    ) -> dict[str, Union[None, pd.DataFrame]]:
        features_dict: dict[str, Any] = {}
        for feature in features:
            features_dict[feature] = None
            if feature in self.config["features"]:
                features_dict[feature] = self.data[symbols][start:end]  # type: ignore[misc]
            elif feature in self.config["tech_indicators"]:
                farg = args.get(feature, [])
                if not isinstance(farg, list):
                    farg = [farg]
                name = (
                    f"{feature}_{"_".join([str(arg) for arg in farg])}"
                    if farg
                    else feature
                )
                print(f"Calculating {name}")
                features_dict[name] = eval(
                    f"Techlib.{feature}(self.data[symbols], *farg)"
                ).loc[start:end] # type: ignore[misc]
        return features_dict


if __name__ == "__main__":
    dl = DataLoader()
    features = dl.load_data(
        "2024-11-11 00:00:00",
        "2024-11-11 23:54:00",
        ["BTC", "ZRX"],
        ["ma", "rsi", "macd"],
        {"ma": 10, "macd": [12, 26]},
    )
