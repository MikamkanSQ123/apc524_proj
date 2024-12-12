import pandas as pd
from typing import Any, Union
from .techlib import Techlib
from pathlib import Path
import ccxt  # type: ignore[import-not-found]


class DataLoader:
    def __init__(self, config: dict[str, Any]):
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
        if base:
            assert base in self.config["features"]
            self.load_from_file(base)

        features_dict: dict[str, Any] = {}
        for feature in features:
            # load data from local files if it already exists in path
            if feature in self.config["features"]:
                self.load_from_file(feature)
                try:
                    features_dict[feature] = self.data[feature][symbols][start:end]  # type: ignore[misc]
                except KeyError:
                    print(f"Error: {symbols} at {start}:{end} not found in local!")
                    features_dict[feature] = None
            # calculate technical indicators
            elif feature in self.config["tech_indicators"]:
                assert base, "Base feature should be provided for technical indicators"
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
        features_not_found += [f for f in features_dict if features_dict[f] is None]

        if features_not_found:
            print(f"Features not found: {features_not_found}")
            # Fetch online sources if data missing
            if self.config["source"] == "ccxt":
                print("Fetching data from ccxt...")
                dict_not_found = ccxtFetcher.load_data(
                    start, end, symbols, features_not_found
                )
                self.data.update(dict_not_found)
                features_dict.update(dict_not_found)
        return features_dict

    def _test(self) -> Union[pd.DataFrame, "pd.Series[Any]"]:
        "A test function to check if the class is working"
        return Techlib.ma(pd.Series(list(range(100))), 10)


class ccxtFetcher(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def is_feasible(exchange: Union[str, ccxt.Exchange], func: str) -> bool:
        """
        Check if the exchange in ccxt supports the function
        Args:
            exchange (Union[str, ccxt.Exchange]): exchange name or a ccxt exchange object (Eg. "binanceus")
            func (str): function name specialized in cxxt (Eg. "fetchOHLCV")
        Returns:
            bool: True if the function is supported
        """
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
        "Generate time range"
        return pd.date_range(start, end, freq=freq).strftime("%Y-%m-%d").tolist()

    @staticmethod
    def get_tick(
        exchange: Union[str, ccxt.Exchange],
        symbol: str,
        date: str,
        timeframe: str = "1m",
        bars: int = 100000,
    ) -> Any:
        """
        Get tick(date) data from ccxt

        Args:
            exchange (Union[str, ccxt.Exchange]): exchange name or a ccxt exchange object (Eg. "binanceus")
            symbol (str): symbol name (Eg. "BTC/USDT")
            date (str): date (Eg. "2022-01-01") / timestamp (Eg. "2022-01-01 00:00:00")
            timeframe (str): timeframe (Eg. "1m")
            bars (int): number of data to fetch (Eg. 100000) (this API has a limit for bars at one time)

        Return:
            Any: DataFrame of tick data
        """
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
        """
        Merge symbols tick data and reframe features

        Args:
            exchange (Union[str, ccxt.Exchange]): exchange name or a ccxt exchange object (Eg. "binanceus")
            symbols (list[str]): list of symbols (Eg. ["BTC/USDT", "ETH/USDT"])
            start (Union[str, pd.Timestamp]): start time
            end (Union[str, pd.Timestamp]): end time
            timeframe (str): timeframe (Eg. "1m")

        Return:
            Any: DataFrame of tick data with all required symbols
        """
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
        """
        Get required features from ccxt and return as a dictionary

        Args:
            start (Union[str, pd.Timestamp]): start time
            end (Union[str, pd.Timestamp]): end time
            symbols (list[str]): list of symbols (Eg. ["BTC/USDT", "ETH/USDT"])
            features (list[str]): list of features (Eg. ["open", "close", "volume"])
            exchange (str): exchange name (Eg. "binanceus")
            timeframe (str): timeframe (Eg. "1m")

        Return:
            Any: dictionary of features key: feature name, value: feature data
        """
        possible_features = ["open", "high", "low", "close", "volume"]
        fts = [f for f in features if f in possible_features]
        df = ccxtFetcher.get_data(exchange, symbols, start, end, timeframe)
        features_dict = {
            col: df.pivot(columns="symbol", values=col)[start:end]  # type: ignore[misc]
            for col in fts
        }
        return features_dict


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
