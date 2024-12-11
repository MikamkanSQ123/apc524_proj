import pandas as pd
from typing import Any, Union
from .techlib import Techlib
from pathlib import Path

path = "./src/simple_backtester/data/feature/"
config = {
    "data_path": path,
    # "tech_indicators": ["ma", "macd", "rsi"],
    "features": [file.name[:-4] for file in Path(path).iterdir() if file.is_file()],
}


class DataLoader:
    def __init__(self, config: dict[str, Any] = config):
        self.config = config
        # Data path should be provided
        assert "data_path" in config
        if not Path(config["data_path"]).exists():
            raise FileNotFoundError(f"Path {config['data_path']} not found")
        print(f'Loading data from {config["data_path"]}')

        # Features should be provided
        if "features" not in config:
            self.config["features"] = [
                file.name[:-4]
                for file in Path(config["data_path"]).iterdir()
                if file.is_file()
            ]
        print(f'Possible target features: {config["features"]}')

        # Technical indicators provided ( default: all ) (这个功能你可以不用但我不能不加)
        if "tech_indicators" not in config:
            self.config["tech_indicators"] = [
                name for name in dir(Techlib) if not name.startswith("__")
            ]

        self.data: dict[str, Any] = {}

    @staticmethod
    def read(file: Union[str, Path]) -> pd.DataFrame:
        if isinstance(file, str):
            file = Path(file)
        return pd.read_csv(file, index_col=0, parse_dates=True)

    def load_from_file(self, name: str) -> Any:
        if name not in self.config["features"]:
            raise ValueError(f"Feature {name} not found in {self.config['data_path']}")
        if name not in self.data:
            path = self.config["data_path"] + name + ".csv"
            self.data[name] = DataLoader.read(path)
        return self.data[name]

    def load_data(
        self,
        start: str,
        end: str,
        symbols: list[str],
        features: list[str],
        args: dict[str, Any] = {},
        base: str = "price",
    ) -> dict[str, Union[None, pd.DataFrame]]:
        assert base in self.config["features"]
        self.load_from_file(base)

        features_dict: dict[str, Any] = {}
        for feature in features:
            features_dict[feature] = None
            # load data from local files if it already exists in path
            if feature in self.config["features"]:
                self.load_from_file(feature)
                features_dict[feature] = self.data[feature][symbols][start:end]  # type: ignore[misc]
            # calculate technical indicators
            elif feature in self.config["tech_indicators"]:
                farg = args.get(feature, [])
                if not isinstance(farg, list):
                    farg = [farg]
                name = (
                    f"{feature}_{'_'.join([str(arg) for arg in farg])}"
                    if farg
                    else feature
                )
                print(f"Calculating {name}")
                features_dict[name] = eval(
                    f"Techlib.{feature}(self.data[base][symbols], *farg)"
                ).loc[start:end]  # type: ignore[misc]
        return features_dict

    def _test(self) -> Union[pd.DataFrame, "pd.Series[Any]"]:
        return Techlib.ma(self.data["BTC"], 10)


if __name__ == "__main__":
    dl = DataLoader()
    features = dl.load_data(
        "2024-11-11 00:00:00",
        "2024-11-11 23:54:00",
        ["BTC", "ZRX"],
        ["ma", "rsi", "macd"],
        {"ma": 10, "macd": [12, 26]},
        base="price",
    )
    features = dl.load_data(
        "2024-11-11 00:00:00",
        "2024-11-11 23:54:00",
        ["BTC", "ZRX"],
        ["price", "return"],
    )
    # print(features['price'].iloc[2:4,:])
