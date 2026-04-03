import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime, time


class SessionFilter:
    ASIA = "Asia"
    LONDON = "London"
    NY = "NY"
    OVERLAP = "Overlap"

    SESSIONS = {
        ASIA: (time(0, 0), time(8, 0)),
        LONDON: (time(8, 0), time(16, 0)),
        NY: (time(13, 0), time(21, 0)),
    }

    @classmethod
    def get_session(cls, utc_time: datetime) -> str:
        t = utc_time.time()
        is_london = cls.SESSIONS[cls.LONDON][0] <= t < cls.SESSIONS[cls.LONDON][1]
        is_ny = cls.SESSIONS[cls.NY][0] <= t < cls.SESSIONS[cls.NY][1]

        if is_london and is_ny:
            return cls.OVERLAP
        elif is_london:
            return cls.LONDON
        elif is_ny:
            return cls.NY
        elif cls.SESSIONS[cls.ASIA][0] <= t < cls.SESSIONS[cls.ASIA][1]:
            return cls.ASIA
        else:
            return "Off hours"


class LiquiditySweepDetector:
    @staticmethod
    def find_sweeps(df: pd.DataFrame, lookback: int = 20) -> List[Dict]:
        sweeps = []
        for i in range(lookback, len(df) - 1):
            current = df.iloc[i]
            prev_highs = df.iloc[i-lookback:i]['high'].max()
            prev_lows = df.iloc[i-lookback:i]['low'].min()

            if current['high'] > prev_highs:
                sweeps.append({
                    'timestamp': df.index[i],
                    'type': 'buy_stop_sweep',
                    'level': prev_highs
                })
            elif current['low'] < prev_lows:
                sweeps.append({
                    'timestamp': df.index[i],
                    'type': 'sell_stop_sweep',
                    'level': prev_lows
                })

        return sweeps[-5:]

    @staticmethod
    def has_recent_sweep(df: pd.DataFrame, bars: int = 5) -> bool:
        if len(df) < bars + 1:
            return False
        recent = df.tail(bars)
        prev_high = df.iloc[:-bars]['high'].max()
        prev_low = df.iloc[:-bars]['low'].min()
        return any(recent['high'] > prev_high) or any(recent['low'] < prev_low)


class HTFBiasDetector:
    @staticmethod
    def get_4h_bias(df: pd.DataFrame) -> str:
        if len(df) < 20:
            return "neutral"

        last_10 = df.tail(10)
        swing_high_val = float(last_10['high'].max())
        swing_low_val = float(last_10['low'].min())
        current_close = float(df['close'].iloc[-1])

        if current_close > swing_high_val:
            return "bullish"
        elif current_close < swing_low_val:
            return "bearish"
        return "neutral"

    @staticmethod
    def get_daily_bias(df: pd.DataFrame) -> str:
        if len(df) < 2:
            return "neutral"

        daily_high = float(df['high'].max())
        daily_low = float(df['low'].min())
        current_close = float(df['close'].iloc[-1])

        if current_close > daily_high * 0.99:
            return "bullish"
        elif current_close < daily_low * 1.01:
            return "bearish"
        return "neutral"


class OrderBlockDetector:
    @staticmethod
    def find_order_blocks(df: pd.DataFrame, n: int = 5) -> List[Dict]:
        blocks = []
        for i in range(len(df) - 1, max(0, len(df) - 50), -1):
            if df.iloc[i]['close'] > df.iloc[i]['open']:
                block_type = "bullish"
            else:
                block_type = "bearish"

            if i > 0:
                if block_type == "bullish" and df.iloc[i-1]['close'] < df.iloc[i-1]['open']:
                    blocks.append({
                        'timestamp': df.index[i],
                        'type': block_type,
                        'high': df.iloc[i]['high'],
                        'low': df.iloc[i]['low']
                    })
                elif block_type == "bearish" and df.iloc[i-1]['close'] > df.iloc[i-1]['open']:
                    blocks.append({
                        'timestamp': df.index[i],
                        'type': block_type,
                        'high': df.iloc[i]['high'],
                        'low': df.iloc[i]['low']
                    })

        return blocks[:n]

    @staticmethod
    def check_overlap(order_blocks: List[Dict], fvg: Dict) -> bool:
        for ob in order_blocks:
            if fvg['type'] == 'bullish':
                if ob['type'] == 'bullish':
                    if ob['high'] >= fvg['low'] and ob['low'] <= fvg['high']:
                        return True
            elif fvg['type'] == 'bearish':
                if ob['type'] == 'bearish':
                    if ob['high'] >= fvg['low'] and ob['low'] <= fvg['high']:
                        return True
        return False


class SMTDivergence:
    @staticmethod
    def detect_divergence(df_mnq: pd.DataFrame, df_mes: pd.DataFrame, lookback: int = 10) -> Optional[Dict]:
        if len(df_mnq) < lookback or len(df_mes) < lookback:
            return None

        mnq_recent = df_mnq.tail(lookback)
        mes_recent = df_mes.tail(lookback)

        mnq_hh = mnq_recent['high'].max()
        mnq_lh = mnq_recent[mnq_recent['high'] < mnq_hh]['high'].max() if len(mnq_recent[mnq_recent['high'] < mnq_hh]) > 0 else mnq_hh
        mnq_hl = mnq_recent['low'].min()
        mnq_ll = mnq_recent[mnq_recent['low'] > mnq_hl]['low'].min() if len(mnq_recent[mnq_recent['low'] > mnq_hl]) > 0 else mnq_hl

        mes_hh = mes_recent['high'].max()
        mes_lh = mes_recent[mes_recent['high'] < mes_hh]['high'].max() if len(mes_recent[mes_recent['high'] < mes_hh]) > 0 else mes_hh
        mes_hl = mes_recent['low'].min()
        mes_ll = mes_recent[mes_recent['low'] > mes_hl]['low'].min() if len(mes_recent[mes_recent['low'] > mes_hl]) > 0 else mes_hl

        mnq_trend = "bullish" if mnq_hh > mnq_lh else "bearish"
        mes_trend = "bullish" if mes_hh > mes_lh else "bearish"

        if mnq_trend != mes_trend:
            return {
                'type': 'divergence',
                'mnq_trend': mnq_trend,
                'mes_trend': mes_trend,
                'direction': 'bullish' if mnq_trend == 'bullish' else 'bearish'
            }

        return None


class SignalScorer:
    @staticmethod
    def calculate_score(
        fvg_present: bool,
        htf_bullish: bool,
        htf_bearish: bool,
        liquidity_sweep: bool,
        order_block_overlap: bool,
        smt_divergence: bool,
        session: str,
        ifvg_present: bool = False
    ) -> int:
        score = 0

        if fvg_present:
            score += 20

        if ifvg_present:
            score += 10

        if htf_bullish or htf_bearish:
            score += 20

        if liquidity_sweep:
            score += 15

        if order_block_overlap:
            score += 15

        if smt_divergence:
            score += 10

        session_quality = {
            SessionFilter.NY: 10,
            SessionFilter.LONDON: 8,
            SessionFilter.OVERLAP: 10,
            SessionFilter.ASIA: 5,
            "Off hours": 0
        }
        score += session_quality.get(session, 0)

        return min(score, 100)

    @staticmethod
    def get_grade(score: int) -> str:
        if score >= 80:
            return "A+"
        elif score >= 60:
            return "A"
        elif score >= 40:
            return "B"
        elif score >= 20:
            return "C"
        return "D"