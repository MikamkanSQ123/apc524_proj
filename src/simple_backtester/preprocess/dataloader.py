import pandas as pd
from typing import Any, Union
from .techlib import Techlib
from pathlib import Path
import ccxt  # type: ignore[import-not-found]

source = "ccxt"
path = f".src/tests/test_data/data/{source}/feature/"
config = {
    "source": source,
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
        # For not found features
        features_not_found = [
            f
            for f in features
            if (f not in self.config["features"])
            and (f not in self.config["tech_indicators"])
        ]
        print(f"Features not found: {features_not_found}")
        if self.config["source"] == "ccxt":
            print("Fetching data from ccxt...")
            dict_not_found = ccxtFetcher.load_data(
                start, end, symbols, features_not_found
            )
            self.data.update(dict_not_found)
            features_dict.update(dict_not_found)
        return features_dict

    def _test(self) -> Union[pd.DataFrame, "pd.Series[Any]"]:
        return Techlib.ma(self.data["BTC"], 10)


class ccxtFetcher(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def is_feasible(exchange: Union[str, ccxt.Exchange], func: str) -> bool:
        try:
            if isinstance(exchange, str):
                exchange = getattr(ccxt, exchange)()
            if exchange.has[func]:
                return True
        except Exception:
            # Handle exceptions (e.g., API errors, unsupported methods)
            return False
        return False

    @staticmethod
    def get_time(
        start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp], freq: str = "D"
    ) -> Any:
        return pd.date_range(start, end, freq=freq).strftime("%Y-%m-%d").tolist()

    @staticmethod
    def get_tick(
        exchange: Union[str, ccxt.Exchange],
        symbol: str,
        date: str,
        timeframe: str = "1m",
        bars: int = 100000,
    ) -> Any:
        if isinstance(exchange, str):
            exchange = getattr(ccxt, exchange)()
        since = exchange.parse8601(f"{date}T00:00:00Z")
        # since = exchange.parse8601(date)
        assert ccxtFetcher.is_feasible(exchange, "fetchOHLCV")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, bars)

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    @staticmethod
    def get_data(
        exchange: Union[str, ccxt.Exchange],
        symbols: list[str],
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        timeframe: str = "1m",
    ) -> Any:
        time = ccxtFetcher.get_time(start, end)
        dfs = pd.DataFrame()
        for t in time:
            for symbol in symbols:
                try:
                    df = ccxtFetcher.get_tick(exchange, symbol, t, timeframe)
                    df["symbol"] = symbol
                    dfs = pd.concat([dfs, df], axis=0)
                except Exception as e:
                    print(f"Error: {e} when running {exchange} {symbol} {t}")
        return dfs

    @staticmethod
    def load_data(
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        symbols: list[str],
        features: list[str],
        exchange: str = "binanceus",
        timeframe: str = "1m",
    ) -> Any:
        possible_features = ["open", "high", "low", "close", "volume"]
        fts = [f for f in features if f in possible_features]
        df = ccxtFetcher.get_data(exchange, symbols, start, end, timeframe)
        features_dict = {
            col: df.pivot(columns="symbol", values=col)[start:end]  # type: ignore[misc]
            for col in fts
        }
        return features_dict


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
    # only if source is ccxt
    features = dl.load_data(
        "2022-01-01",
        "2022-01-02",
        ["1INCH/USDT", "1INCH/USD"],
        ["open", "close"],
        base="close",
    )
    # print(features['price'].iloc[2:4,:])
