from typing import Optional, Dict, List
import pandas as pd
from datetime import datetime

from src.data.data_manager import DataManager
from src.signals.fvg_detector import FVGDetector
from src.signals.signal_components import (
    SessionFilter,
    HTFBiasDetector,
    LiquiditySweepDetector,
    OrderBlockDetector,
    SMTDivergence,
    SignalScorer
)


class SignalEngine:
    def __init__(self, db_path: str = "data/trading.db"):
        self.data_manager = DataManager(db_path)
        self.fvg_detector = FVGDetector()

    def analyze(self, symbol: str = "MNQ1!", exchange: str = "CME", interval: str = "1H", n_bars: int = 200) -> Dict:
        df = self.data_manager.get_data(symbol, exchange, interval, n_bars)
        if df.empty:
            return {'error': 'No data available'}

        df = self.fvg_detector.detect(df)

        current_time = df.index[-1] if hasattr(df.index[-1], 'hour') else datetime.now()
        session = SessionFilter.get_session(current_time)

        htf_4h = self.data_manager.get_data(symbol, exchange, "4H", 50)
        if not htf_4h.empty:
            htf_4h = self.fvg_detector.detect(htf_4h)
            bias_4h = HTFBiasDetector.get_4h_bias(htf_4h)
        else:
            bias_4h = "neutral"

        htf_daily = self.data_manager.get_data(symbol, exchange, "1D", 30)
        if not htf_daily.empty:
            htf_daily = self.fvg_detector.detect(htf_daily)
            bias_daily = HTFBiasDetector.get_daily_bias(htf_daily)
        else:
            bias_daily = "neutral"

        latest_fvg = self.fvg_detector.get_latest(df, n=3)
        latest_fvg = latest_fvg[-1] if latest_fvg else None

        liquidity_sweep = LiquiditySweepDetector.has_recent_sweep(df)
        order_blocks = OrderBlockDetector.find_order_blocks(df)
        ob_overlap = False

        smt_div = None
        if latest_fvg:
            df_mes = self.data_manager.get_data("MES1!", exchange, interval, n_bars)
            if not df_mes.empty:
                df_mes = self.fvg_detector.detect(df_mes)
                smt_div = SMTDivergence.detect_divergence(df, df_mes)
                if smt_div:
                    if latest_fvg['type'] == 'bullish' and smt_div['direction'] == 'bullish':
                        ob_overlap = OrderBlockDetector.check_overlap(order_blocks, latest_fvg)
                    elif latest_fvg['type'] == 'bearish' and smt_div['direction'] == 'bearish':
                        ob_overlap = OrderBlockDetector.check_overlap(order_blocks, latest_fvg)

        fvg_present = latest_fvg is not None
        score = SignalScorer.calculate_score(
            fvg_present=fvg_present,
            htf_bullish=bias_4h == "bullish" or bias_daily == "bullish",
            htf_bearish=bias_4h == "bearish" or bias_daily == "bearish",
            liquidity_sweep=liquidity_sweep,
            order_block_overlap=ob_overlap,
            smt_divergence=smt_div is not None,
            session=session
        )

        result = {
            'timestamp': current_time,
            'symbol': symbol,
            'session': session,
            'htf_4h_bias': bias_4h,
            'htf_daily_bias': bias_daily,
            'latest_fvg': latest_fvg,
            'liquidity_sweep': liquidity_sweep,
            'order_blocks': order_blocks[:3],
            'ob_overlap': ob_overlap,
            'smt_divergence': smt_div,
            'score': score,
            'grade': SignalScorer.get_grade(score),
            'ohlcv': df.tail(50).reset_index().to_dict('records')
        }

        if latest_fvg and score >= 60:
            self.data_manager.save_signal({
                'timestamp': current_time,
                'symbol': symbol,
                'direction': latest_fvg['type'],
                'fvg_type': latest_fvg['type'],
                'fvg_high': latest_fvg.get('high'),
                'fvg_low': latest_fvg.get('low'),
                'htf_bias': bias_4h,
                'liquidity_sweep': int(liquidity_sweep),
                'order_block': int(ob_overlap),
                'smt_divergence': int(smt_div is not None),
                'session': session,
                'score': score,
                'filled': 0,
                'profit': 0.0
            })

        return result

    def run_backtest(self, symbol: str = "MNQ1!", exchange: str = "CME", interval: str = "1H",
                     start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict:
        # Fetch historical price data
        df = self.data_manager.get_data(symbol, exchange, interval, 500)
        if df.empty:
            return {'error': 'No historical data available'}

        # Filter by date range if provided
        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.Timestamp(end_date)]

        if len(df) < 50:
            return {'error': 'Not enough data for backtest'}

        # Also fetch correlated instrument for SMT
        smt_symbol = "MES1!" if symbol == "MNQ1!" else "MNQ1!"
        df_smt = self.data_manager.get_data(smt_symbol, exchange, interval, 500)

        # Fetch HTF data
        htf_4h = self.data_manager.get_data(symbol, exchange, "4H", 50)
        if not htf_4h.empty:
            htf_4h = self.fvg_detector.detect(htf_4h)
            bias_4h = HTFBiasDetector.get_4h_bias(htf_4h)
        else:
            bias_4h = "neutral"

        htf_daily = self.data_manager.get_data(symbol, exchange, "1D", 30)
        if not htf_daily.empty:
            htf_daily = self.fvg_detector.detect(htf_daily)
            bias_daily = HTFBiasDetector.get_daily_bias(htf_daily)
        else:
            bias_daily = "neutral"

        # Walk forward through the data generating signals
        window = 50
        signals = []
        for i in range(window, len(df)):
            window_df = df.iloc[i - window:i + 1].copy()
            window_df = self.fvg_detector.detect(window_df)

            latest_fvg = self.fvg_detector.get_latest(window_df, n=3)
            latest_fvg = latest_fvg[-1] if latest_fvg else None
            if not latest_fvg:
                continue

            current_time = window_df.index[-1]
            session = SessionFilter.get_session(current_time) if hasattr(current_time, 'hour') else "Off hours"

            liquidity_sweep = LiquiditySweepDetector.has_recent_sweep(window_df)
            order_blocks = OrderBlockDetector.find_order_blocks(window_df)
            ob_overlap = OrderBlockDetector.check_overlap(order_blocks, latest_fvg)

            smt_div = None
            if not df_smt.empty and len(df_smt) > i:
                smt_window = df_smt.iloc[max(0, i - window):i + 1]
                if len(smt_window) >= 10:
                    smt_window = self.fvg_detector.detect(smt_window)
                    smt_div = SMTDivergence.detect_divergence(window_df, smt_window)

            score = SignalScorer.calculate_score(
                fvg_present=True,
                htf_bullish=bias_4h == "bullish" or bias_daily == "bullish",
                htf_bearish=bias_4h == "bearish" or bias_daily == "bearish",
                liquidity_sweep=liquidity_sweep,
                order_block_overlap=ob_overlap,
                smt_divergence=smt_div is not None,
                session=session
            )

            if score < 60:
                continue

            # Simple profit estimation: check if price moved in FVG direction
            # over the next few bars (if available)
            profit = 0.0
            if i + 5 < len(df):
                entry_price = df.iloc[i]['close']
                exit_price = df.iloc[i + 5]['close']
                if latest_fvg['type'] == 'bullish':
                    profit = exit_price - entry_price
                else:
                    profit = entry_price - exit_price

            signals.append({
                'timestamp': current_time,
                'symbol': symbol,
                'direction': latest_fvg['type'],
                'fvg_type': latest_fvg['type'],
                'session': session,
                'score': score,
                'profit': round(profit, 2),
            })

        if not signals:
            return {'error': 'No signals generated during backtest period'}

        signals_df = pd.DataFrame(signals)
        total = len(signals_df)
        won = len(signals_df[signals_df['profit'] > 0])
        win_rate = (won / total * 100) if total > 0 else 0

        return {
            'total_signals': total,
            'winning_trades': won,
            'losing_trades': total - won,
            'win_rate': win_rate,
            'avg_profit': round(signals_df['profit'].mean(), 2) if total > 0 else 0,
            'signals': signals_df.to_dict('records')
        }

    def get_fvg_fill_stats(self) -> Dict:
        signals = self.data_manager.get_signals()
        if signals.empty:
            return {'fill_rate': 0, 'total': 0}

        total = len(signals)
        filled = len(signals[signals['filled'] == 1])
        
        session_stats = signals.groupby('session').agg({
            'id': 'count',
            'filled': 'sum',
            'profit': 'mean'
        }).to_dict('records')

        return {
            'fill_rate': (filled / total * 100) if total > 0 else 0,
            'total': total,
            'filled': filled,
            'by_session': session_stats
        }