import pandas as pd
from typing import Any, Union
import pandas_market_calendars as mcal

class Utils(object):
    @staticmethod
    def get_trading_days(start: Union[str,pd.Timestamp], end: Union[str,pd.Timestamp], exchange: str = "NYSE") -> pd.DatetimeIndex:
        nyse = mcal.get_calendar(exchange)
        return nyse.valid_days(start_date=start, end_date=end)

    @staticmethod
    def get_trading_days_count(start: Union[str,pd.Timestamp], end: Union[str,pd.Timestamp], exchange: str = "NYSE") -> int:
        return len(Utils.get_trading_days(start, end, exchange))

    @staticmethod
    def get_trading_days_list(start: str, end: str, exchange: str = "NYSE") -> list[str]:
        return Utils.get_trading_days(start, end, exchange).strftime("%Y-%m-%d").tolist()

    @staticmethod
    def is_trade_day(date: Union[str,pd.Timestamp], exchange: str = "NYSE") -> bool:
        if isinstance(date, pd.Timestamp):
            date = date.strftime("%Y-%m-%d")
        return date in Utils.get_trading_days_list(date, date, exchange)
    
    @staticmethod
    def get_previous_trade_day(date: Union[str,pd.Timestamp], exchange: str = "NYSE") -> str:
        while True:
            date = pd.to_datetime(date) - pd.Timedelta(days=1)
            if Utils.is_trade_day(date, exchange):
                return date.strftime("%Y-%m-%d")
    