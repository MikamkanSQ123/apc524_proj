import pandas as pd
from typing import Any, Union
from .techlib import Techlib
from pathlib import Path
from .source import Source, localSource, ccxtSource


class DataLoader:
    source: Source

    def __init__(self, config: dict[str, Any]):
        self.config = config
        # Data path should be provided
        assert "data_path" in config
        if not Path(config["data_path"]).exists():
            raise FileNotFoundError(f"Path {config['data_path']} not found")
        print(f'Loading data from {config["data_path"]}')

        # Get source
        source = config.get("source", "local")
        if source == "ccxt":
            self.source = ccxtSource()
        else:
            self.source = localSource()

        # Features should be provided
        if "features" not in config:
            self.config["features"] = [
                file.name[:-4]
                for file in Path(config["data_path"]).iterdir()
                if file.is_file()
            ]
        print(f'Possible target features: {config["features"]}')

        # Technical indicators provided ( default: all ) ( optional )
        if "tech_indicators" not in config:
            self.config["tech_indicators"] = [
                name for name in dir(Techlib) if not name.startswith("__")
            ]

        # Data storage
        self.data: dict[str, Any] = {}

    @staticmethod
    def read(file: Union[str, Path]) -> pd.DataFrame:
        "Read csv files, with index and time transformation"
        if isinstance(file, str):
            file = Path(file)
        return pd.read_csv(file, index_col=0, parse_dates=True)

    def load_from_file(self, name: str) -> Any:
        "Load data from local files"
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
        base: str = "",
    ) -> dict[str, Union[None, pd.DataFrame]]:
        """
        Load data from local files/online sources or calculate technical indicators.

        Args:
            start(str): start time
            end(str): end time
            symbols(list[str]): list of symbols (Eg. ["BTC", "ETH"])
            features(list[str]): list of features (Eg. ["close", "return", "ma"])
            args(dict[str, Any]): arguments for technical indicators (Eg. {"ma": 10})
            base(str): base feature for technical indicators (Eg. "close")

        Return:
            dict[str, Union[None, pd.DataFrame]]: dictionary of features key: feature name, value: feature data
        """
        basic_features, tech_indicators, features_not_found = [], [], []
        for feature in features:
            if feature in self.config["features"]:
                basic_features.append(feature)
            elif feature in self.config["tech_indicators"]:
                tech_indicators.append(feature)
            else:
                features_not_found.append(feature)

        features_dict: dict[str, Any] = {}
        # load data from local files if it already exists in path
        for feature in basic_features:
            self.load_from_file(feature)
            try:
                features_dict[feature] = self.data[feature][symbols][start:end]  # type: ignore[misc]
            except KeyError:
                print(f"Error: {symbols} at {start}:{end} not found in local!")
                features_dict[feature] = None
        # For not found features
        features_not_found += [f for f in features_dict if features_dict[f] is None]
        dict_not_found = self.source.load_data(start, end, symbols, features_not_found)
        self.data.update(dict_not_found)
        features_dict.update(dict_not_found)
        # calculate technical indicators
        if tech_indicators:
            assert base, "Base feature should be provided for technical indicators"
            assert (
                base in features_dict and features_dict[base] is not None
            ), f"Base feature {base} not found"
            for feature in tech_indicators:
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
        "A test function to check if the class is working"
        return Techlib.ma(pd.Series(list(range(100))), 10)


if __name__ == "__main__":
    source = "ccxt"
    path = f"tests/test_data/data/{source}/feature/"
    config = {
        "source": source,
        "data_path": path,
        # "tech_indicators": ["ma", "macd", "rsi"],
        "features": [file.name[:-4] for file in Path(path).iterdir() if file.is_file()],
    }
    dl = DataLoader(config=config)
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
    # only if source is ccxt
    features = dl.load_data(
        "2022-01-01",
        "2022-01-02",
        ["1INCH/USDT", "1INCH/USD"],
        ["open", "close"],
        base="close",
    )
    # print(features['price'].iloc[2:4,:])
