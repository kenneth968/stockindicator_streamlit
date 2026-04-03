from typing import Optional, Dict, List
from itertools import groupby
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
    # SMT pairs: Nasdaq instruments pair with S&P instruments
    SMT_PAIRS = {
        "MNQ1!": "MES1!",
        "NQ1!": "ES1!",
        "MES1!": "MNQ1!",
        "ES1!": "NQ1!",
    }

    def __init__(self, db_path: str = "data/trading.db"):
        self.data_manager = DataManager(db_path)
        self.fvg_detector = FVGDetector()

    def analyze(self, symbol: str = "MNQ1!", exchange: str = "CME", interval: str = "1H", n_bars: int = 200) -> Dict:
        df = self.data_manager.get_data(symbol, exchange, interval, n_bars)
        if df.empty:
            return {'error': 'No data available'}

        # Normalize timezone-aware index to naive for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = self.fvg_detector.detect(df)

        current_time = df.index[-1] if hasattr(df.index[-1], 'hour') else datetime.now()
        session = SessionFilter.get_session(current_time)

        htf_4h = self.data_manager.get_data(symbol, exchange, "4H", 50)
        if htf_4h is not None and not htf_4h.empty:
            if htf_4h.index.tz is not None:
                htf_4h.index = htf_4h.index.tz_localize(None)
            htf_4h = self.fvg_detector.detect(htf_4h)
            bias_4h = HTFBiasDetector.get_4h_bias(htf_4h)
        else:
            bias_4h = "neutral"

        htf_daily = self.data_manager.get_data(symbol, exchange, "1D", 30)
        if htf_daily is not None and not htf_daily.empty:
            if htf_daily.index.tz is not None:
                htf_daily.index = htf_daily.index.tz_localize(None)
            htf_daily = self.fvg_detector.detect(htf_daily)
            bias_daily = HTFBiasDetector.get_daily_bias(htf_daily)
        else:
            bias_daily = "neutral"

        all_fvgs = self.fvg_detector.get_latest(df, n=5)
        latest_fvg = all_fvgs[-1] if all_fvgs else None

        ifvg = self.fvg_detector.check_ifvg(df)

        liquidity_sweeps = LiquiditySweepDetector.find_sweeps(df)
        liquidity_sweep = LiquiditySweepDetector.has_recent_sweep(df)
        order_blocks = OrderBlockDetector.find_order_blocks(df)
        ob_overlap = False

        smt_div = None
        smt_symbol = self.SMT_PAIRS.get(symbol, "MES1!")
        if latest_fvg:
            df_mes = self.data_manager.get_data(smt_symbol, exchange, interval, n_bars)
            if df_mes is not None and not df_mes.empty:
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
            session=session,
            ifvg_present=ifvg is not None
        )

        result = {
            'timestamp': current_time,
            'symbol': symbol,
            'session': session,
            'htf_4h_bias': bias_4h,
            'htf_daily_bias': bias_daily,
            'latest_fvg': latest_fvg,
            'all_fvgs': all_fvgs,
            'ifvg': ifvg,
            'liquidity_sweep': liquidity_sweep,
            'liquidity_sweeps': liquidity_sweeps,
            'order_blocks': order_blocks[:3],
            'ob_overlap': ob_overlap,
            'smt_divergence': smt_div,
            'score': score,
            'grade': SignalScorer.get_grade(score),
            'ohlcv': df.tail(50).reset_index().to_dict('records')
        }

        if latest_fvg and score >= 40:
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
        # Determine how many bars to fetch based on interval and date range
        interval_hours = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "30m": 0.5,
                          "1H": 1, "2H": 2, "4H": 4, "1D": 24, "1W": 168, "1M": 720}
        hours_per_bar = interval_hours.get(interval, 1)
        if start_date:
            days_back = (datetime.now() - start_date).days
        else:
            days_back = 365
        estimated_bars = min(int((days_back * 24) / hours_per_bar) + 100, 5000)

        # Fetch historical price data
        df = self.data_manager.get_data(symbol, exchange, interval, estimated_bars)
        if df is None or df.empty:
            return {'error': f'No historical data available for {symbol} ({interval}). Check data source connection.'}

        # Normalize index to timezone-naive for comparison
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        if len(df) < 60:
            return {'error': f'Not enough data for backtest ({len(df)} bars available, need at least 60)'}

        # Also fetch correlated instrument for SMT
        smt_symbol = self.SMT_PAIRS.get(symbol, "MES1!")
        df_smt = self.data_manager.get_data(smt_symbol, exchange, interval, estimated_bars)
        if df_smt is not None and not df_smt.empty and df_smt.index.tz is not None:
            df_smt.index = df_smt.index.tz_localize(None)
        if df_smt is None:
            df_smt = pd.DataFrame()

        # Fetch HTF data for per-bar bias computation (avoid look-ahead)
        htf_4h_raw = self.data_manager.get_data(symbol, exchange, "4H", 200)
        if htf_4h_raw is not None and not htf_4h_raw.empty:
            if htf_4h_raw.index.tz is not None:
                htf_4h_raw.index = htf_4h_raw.index.tz_localize(None)
        else:
            htf_4h_raw = None

        htf_daily_raw = self.data_manager.get_data(symbol, exchange, "1D", 200)
        if htf_daily_raw is not None and not htf_daily_raw.empty:
            if htf_daily_raw.index.tz is not None:
                htf_daily_raw.index = htf_daily_raw.index.tz_localize(None)
        else:
            htf_daily_raw = None

        # Walk forward through the data generating signals
        window = 50
        signals = []
        # Cache HTF bias - recompute every 10 bars to avoid O(n^2) FVG detection
        cached_bias_4h = "neutral"
        cached_bias_daily = "neutral"
        htf_recompute_interval = 10

        for i in range(window, len(df)):
            window_df = df.iloc[i - window:i + 1].copy()
            window_df = self.fvg_detector.detect(window_df)

            latest_fvg = self.fvg_detector.get_latest(window_df, n=3)
            latest_fvg = latest_fvg[-1] if latest_fvg else None
            if not latest_fvg:
                continue

            current_time = window_df.index[-1]
            session = SessionFilter.get_session(current_time) if hasattr(current_time, 'hour') else "Off hours"

            # Recompute HTF bias periodically (avoids running FVG detect every bar)
            if (i - window) % htf_recompute_interval == 0:
                if htf_4h_raw is not None:
                    htf_4h_slice = htf_4h_raw[htf_4h_raw.index <= current_time]
                    if len(htf_4h_slice) >= 20:
                        htf_4h_slice = self.fvg_detector.detect(htf_4h_slice)
                        cached_bias_4h = HTFBiasDetector.get_4h_bias(htf_4h_slice)

                if htf_daily_raw is not None:
                    htf_daily_slice = htf_daily_raw[htf_daily_raw.index <= current_time]
                    if len(htf_daily_slice) >= 2:
                        htf_daily_slice = self.fvg_detector.detect(htf_daily_slice)
                        cached_bias_daily = HTFBiasDetector.get_daily_bias(htf_daily_slice)

            bias_4h = cached_bias_4h
            bias_daily = cached_bias_daily

            liquidity_sweep = LiquiditySweepDetector.has_recent_sweep(window_df)
            order_blocks = OrderBlockDetector.find_order_blocks(window_df)
            ob_overlap = OrderBlockDetector.check_overlap(order_blocks, latest_fvg)

            smt_div = None
            if not df_smt.empty and len(df_smt) > i:
                smt_window = df_smt.iloc[max(0, i - window):i + 1]
                if len(smt_window) >= 10:
                    smt_window = self.fvg_detector.detect(smt_window)
                    smt_div = SMTDivergence.detect_divergence(window_df, smt_window)

            ifvg = self.fvg_detector.check_ifvg(window_df)

            score = SignalScorer.calculate_score(
                fvg_present=True,
                htf_bullish=bias_4h == "bullish" or bias_daily == "bullish",
                htf_bearish=bias_4h == "bearish" or bias_daily == "bearish",
                liquidity_sweep=liquidity_sweep,
                order_block_overlap=ob_overlap,
                smt_divergence=smt_div is not None,
                session=session,
                ifvg_present=ifvg is not None
            )

            if score < 40:
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
            return {'error': f'No signals generated during backtest period ({len(df)} bars analyzed)'}

        signals_df = pd.DataFrame(signals)

        # Filter signals by date range
        if start_date is not None:
            signals_df = signals_df[signals_df['timestamp'] >= pd.Timestamp(start_date)]
        if end_date is not None:
            signals_df = signals_df[signals_df['timestamp'] <= pd.Timestamp(end_date)]

        if signals_df.empty:
            return {'error': 'No signals in selected date range'}
        total = len(signals_df)
        won = len(signals_df[signals_df['profit'] > 0])
        lost = len(signals_df[signals_df['profit'] <= 0])
        win_rate = (won / total * 100) if total > 0 else 0

        profits = signals_df['profit']
        avg_win = float(profits[profits > 0].mean()) if won > 0 else 0.0
        avg_loss = float(profits[profits <= 0].mean()) if lost > 0 else 0.0
        profit_factor = abs(avg_win * won / (avg_loss * lost)) if lost > 0 and avg_loss != 0 else float('inf') if won > 0 else 0.0

        # Cumulative P&L and drawdown
        cum_pnl = profits.cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Sharpe ratio (annualized, assuming ~252 trading days)
        if profits.std() > 0:
            sharpe = float((profits.mean() / profits.std()) * (252 ** 0.5))
        else:
            sharpe = 0.0

        # Consecutive wins/losses
        is_win = (profits > 0).astype(int)
        max_consec_wins = max((len(list(g)) for k, g in groupby(is_win) if k == 1), default=0)
        max_consec_losses = max((len(list(g)) for k, g in groupby(is_win) if k == 0), default=0)

        return {
            'total_signals': total,
            'winning_trades': won,
            'losing_trades': lost,
            'win_rate': round(win_rate, 1),
            'avg_profit': round(float(profits.mean()), 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
            'total_pnl': round(float(cum_pnl.iloc[-1]), 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_consec_wins': max_consec_wins,
            'max_consec_losses': max_consec_losses,
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