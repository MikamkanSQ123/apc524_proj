import pandas as pd
from typing import Any, Union
import ccxt  # type: ignore[import-not-found]
from abc import ABC, abstractmethod


class Source(ABC):
    name: str

    @abstractmethod
    def __init__(self) -> None:
        # Some sources need to be initialized
        pass

    @abstractmethod
    def load_data(
        self, start: str, end: str, symbols: list[str], features: list[str]
    ) -> dict[str, Union[None, pd.DataFrame]]:
        pass


class localSource(Source):  # No other external source for local data
    name = "local"

    def __init__(self) -> None:
        pass

    def load_data(
        self, start: str, end: str, symbols: list[str], features: list[str]
    ) -> dict[str, Union[None, pd.DataFrame]]:
        return {feature: None for feature in features}


class ccxtSource(Source):
    name = "ccxt"

    def __init__(self, exchange: str = "binanceus", timeframe: str = "1m") -> None:
        self.exchange = exchange
        self.timeframe = timeframe

    def load_data(
        self, start: str, end: str, symbols: list[str], features: list[str]
    ) -> Any:
        if not features:
            return {}
        return ccxtFetcher.load_data(
            start, end, symbols, features, self.exchange, self.timeframe
        )


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
