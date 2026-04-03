from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


class FVGDetector:
    BULLISH = "bullish"
    BEARISH = "bearish"
    NONE = "none"

    @staticmethod
    def detect(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['fvg_type'] = FVGDetector.NONE
        df['fvg_high'] = np.nan
        df['fvg_low'] = np.nan

        for i in range(1, len(df) - 1):
            prev = df.iloc[i - 1]
            mid = df.iloc[i]
            next_bar = df.iloc[i + 1]

            mid_low = mid['low']
            mid_high = mid['high']
            prev_high = prev['high']
            prev_low = prev['low']
            next_high = next_bar['high']
            next_low = next_bar['low']

            if mid_low > prev_high:
                df.at[df.index[i], 'fvg_type'] = FVGDetector.BULLISH
                df.at[df.index[i], 'fvg_high'] = min(prev_high, next_high)
                df.at[df.index[i], 'fvg_low'] = mid_low
            elif mid_high < prev_low:
                df.at[df.index[i], 'fvg_type'] = FVGDetector.BEARISH
                df.at[df.index[i], 'fvg_high'] = mid_high
                df.at[df.index[i], 'fvg_low'] = max(prev_low, next_low)

        return df

    @staticmethod
    def get_latest(df: pd.DataFrame, n: int = 5) -> List[Dict]:
        fvg_df = df[df['fvg_type'] != FVGDetector.NONE].tail(n)
        return [
            {
                'timestamp': idx,
                'type': row['fvg_type'],
                'high': row['fvg_high'],
                'low': row['fvg_low']
            }
            for idx, row in fvg_df.iterrows()
        ]

    @staticmethod
    def is_filled(fvg: Dict, current_price: float, direction: str) -> bool:
        if direction == FVGDetector.BULLISH:
            return current_price <= fvg['high']
        else:
            return current_price >= fvg['low']

    @staticmethod
    def check_ifvg(df: pd.DataFrame, lookback: int = 3) -> Optional[Dict]:
        fvgs = df[df['fvg_type'] != FVGDetector.NONE].tail(lookback)
        if len(fvgs) < 2:
            return None

        last_fvg = fvgs.iloc[-1]
        prev_fvg = fvgs.iloc[-2]

        if last_fvg['fvg_type'] == prev_fvg['fvg_type']:
            return None

        return {
            'type': 'ifvg',
            'direction': 'bullish' if last_fvg['fvg_type'] == FVGDetector.BEARISH else 'bearish',
            'timestamp': last_fvg.name,
            'high': min(last_fvg['fvg_high'], prev_fvg['fvg_high']),
            'low': max(last_fvg['fvg_low'], prev_fvg['fvg_low'])
        }