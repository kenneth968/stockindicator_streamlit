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
        if start_date is None:
            start_date = datetime.now() - pd.Timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        all_signals = self.data_manager.get_signals(symbol, start_date)

        if all_signals.empty:
            return {'error': 'No historical signals found'}

        total = len(all_signals)
        won = len(all_signals[all_signals['profit'] > 0])
        win_rate = (won / total * 100) if total > 0 else 0

        return {
            'total_signals': total,
            'winning_trades': won,
            'losing_trades': total - won,
            'win_rate': win_rate,
            'avg_profit': all_signals['profit'].mean() if total > 0 else 0,
            'signals': all_signals.to_dict('records')
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