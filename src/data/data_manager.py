import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)
Base = declarative_base()


class OHLCVCache(Base):
    __tablename__ = 'ohlcv_cache'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    exchange = Column(String(20))
    interval = Column(String(10))
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)


class SignalRecord(Base):
    __tablename__ = 'signal_records'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True)
    symbol = Column(String(20))
    direction = Column(String(10))
    fvg_type = Column(String(20))
    fvg_high = Column(Float)
    fvg_low = Column(Float)
    htf_bias = Column(String(10))
    liquidity_sweep = Column(Integer)
    order_block = Column(Integer)
    smt_divergence = Column(Integer)
    session = Column(String(20))
    score = Column(Integer)
    filled = Column(Integer)
    profit = Column(Float)


class DataManager:
    def __init__(self, db_path: str = "data/trading.db", max_retries: int = 3):
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.max_retries = max_retries
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._tv = None

    def _get_tv_datafeed(self):
        if self._tv is None:
            try:
                from tvdatafeed import TvDatafeed, Interval
                self._tv = TvDatafeed()
                self._tv.Interval = Interval
            except Exception as e:
                logger.warning(f"tvdatafeed not available: {e}")
                return None
        return self._tv

    @staticmethod
    def _convert_interval(interval_str: str):
        from tvdatafeed import Interval
        mapping = {
            "1m": Interval.in_1_minute,
            "5m": Interval.in_5_minute,
            "15m": Interval.in_15_minute,
            "30m": Interval.in_30_minute,
            "1H": Interval.in_1_hour,
            "2H": Interval.in_2_hour,
            "3H": Interval.in_3_hour,
            "4H": Interval.in_4_hour,
            "1D": Interval.in_daily,
            "1W": Interval.in_weekly,
            "1M": Interval.in_monthly,
        }
        return mapping.get(interval_str, Interval.in_1_hour)

    def _fetch_with_retry(self, symbol: str, exchange: str, interval: str, n_bars: int) -> Optional[pd.DataFrame]:
        tv = self._get_tv_datafeed()
        if tv is None:
            return None

        interval_enum = self._convert_interval(interval)

        for attempt in range(self.max_retries):
            try:
                df = tv.get_hist(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval_enum,
                    n_bars=n_bars
                )
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def get_data(self, symbol: str, exchange: str = "CME", interval: str = "1H", n_bars: int = 500) -> pd.DataFrame:
        cached = self._get_cached_data(symbol, interval, n_bars)
        if cached is not None and len(cached) >= n_bars * 0.8:
            logger.info(f"Using cached data for {symbol}")
            return cached

        fresh = self._fetch_with_retry(symbol, exchange, interval, n_bars)
        if fresh is not None:
            self._save_to_cache(fresh, symbol, exchange, interval)
            return fresh

        if cached is not None:
            logger.warning(f"Using stale cache for {symbol}")
            return cached

        return pd.DataFrame()

    def _get_cached_data(self, symbol: str, interval: str, n_bars: int) -> Optional[pd.DataFrame]:
        session = self.Session()
        try:
            query = f"""
                SELECT timestamp, open, high, low, close, volume 
                FROM ohlcv_cache 
                WHERE symbol = '{symbol}' AND interval = '{interval}'
                ORDER BY timestamp DESC 
                LIMIT {n_bars}
            """
            df = pd.read_sql(query, self.engine.connect())
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').set_index('timestamp')
            return df
        finally:
            session.close()

    def _save_to_cache(self, df: pd.DataFrame, symbol: str, exchange: str, interval: str):
        session = self.Session()
        try:
            df = df.reset_index()
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})
            df['symbol'] = symbol
            df['exchange'] = exchange
            df['interval'] = interval
            df.to_sql('ohlcv_cache', self.engine, if_exists='append', index=False)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
        finally:
            session.close()

    def save_signal(self, signal_data: dict):
        session = self.Session()
        try:
            record = SignalRecord(**signal_data)
            session.add(record)
            session.commit()
        except Exception as e:
            logger.warning(f"Signal save failed: {e}")
        finally:
            session.close()

    def get_signals(self, symbol: Optional[str] = None, start_date: Optional[datetime] = None) -> pd.DataFrame:
        session = self.Session()
        try:
            query = "SELECT * FROM signal_records"
            conditions = []
            if symbol:
                conditions.append(f"symbol = '{symbol}'")
            if start_date:
                conditions.append(f"timestamp >= '{start_date.isoformat()}'")
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC"
            df = pd.read_sql(query, self.engine.connect())
            return df
        finally:
            session.close()