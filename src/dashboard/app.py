import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="SMT Trading Signals", layout="wide", page_icon="📈")


def init_session_state():
    if 'signal_engine' not in st.session_state:
        from src.signals.signal_engine import SignalEngine
        st.session_state.signal_engine = SignalEngine()
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True


def plot_candlestick(df: pd.DataFrame, fvgs: list = None) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLCV',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))

    if fvgs:
        for fvg in fvgs:
            if pd.isna(fvg.get('high')) or pd.isna(fvg.get('low')):
                continue
            color = '#26a69a' if fvg['type'] == 'bullish' else '#ef5350'
            fig.add_hrect(
                y0=fvg['low'],
                y1=fvg['high'],
                line_width=0,
                fillcolor=color,
                opacity=0.2,
                annotation_text=f"FVG {fvg['type']}",
                annotation_position="top left"
            )

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def render_live_tab():
    st.header("📡 Live Signals")

    col1, col2 = st.columns([3, 1])

    with col1:
        symbol = st.selectbox("Symbol", ["MNQ1!", "MES1!"], index=0)
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "1H", "4H"], index=3)

    with col2:
        if st.button("🔄 Refresh", type="primary"):
            st.session_state.last_update = None

    if st.session_state.auto_refresh:
        refresh_interval = st.slider("Auto-refresh (seconds)", 30, 300, 60)

    with st.spinner("Fetching data..."):
        result = st.session_state.signal_engine.analyze(symbol=symbol, interval=interval, n_bars=100)

    if 'error' in result:
        st.error(result['error'])
        return

    st.divider()

    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
    with score_col1:
        st.metric("A+ Score", result['score'], result['grade'])
    with score_col2:
        st.metric("Session", result['session'])
    with score_col3:
        st.metric("4H Bias", result['htf_4h_bias'])
    with score_col4:
        st.metric("Daily Bias", result['htf_daily_bias'])

    st.divider()

    check_col1, check_col2, check_col3 = st.columns(3)
    with check_col1:
        st.subheader("✅ Signal Checklist")
        fvg_check = "✅" if result['latest_fvg'] else "❌"
        st.write(f"{fvg_check} FVG Present")
        st.write(f"{'✅' if result['htf_4h_bias'] != 'neutral' else '❌'} HTF Aligned")
        st.write(f"{'✅' if result['liquidity_sweep'] else '❌'} Liquidity Sweep")

    with check_col2:
        st.subheader("🔍 Additional")
        ob_check = "✅" if result['ob_overlap'] else "❌"
        st.write(f"{ob_check} OB-FVG Overlap")
        smt_check = "✅" if result['smt_divergence'] else "❌"
        st.write(f"{smt_check} SMT Divergence")

    with check_col3:
        st.subheader("📊 Order Blocks")
        for i, ob in enumerate(result['order_blocks'][:3]):
            st.write(f"OB {i+1}: {ob['type']} {ob.get('high', 0):.0f}")

    st.divider()

    if result['latest_fvg']:
        fvg = result['latest_fvg']
        fvg_col1, fvg_col2, fvg_col3 = st.columns(3)
        with fvg_col1:
            st.metric("FVG Type", fvg['type'].upper())
        with fvg_col2:
            st.metric("FVG High", f"{fvg.get('high', 0):.2f}")
        with fvg_col3:
            st.metric("FVG Low", f"{fvg.get('low', 0):.2f}")

    st.divider()

    ohlcv_df = pd.DataFrame(result.get('ohlcv', []))
    if not ohlcv_df.empty:
        if 'datetime' in ohlcv_df.columns:
            ohlcv_df.set_index('datetime', inplace=True)
        elif 'timestamp' in ohlcv_df.columns:
            ohlcv_df.set_index('timestamp', inplace=True)

        fvgs_for_chart = [result['latest_fvg']] if result['latest_fvg'] else []
        fig = plot_candlestick(ohlcv_df, fvgs_for_chart)
        st.plotly_chart(fig, use_container_width=True)


def render_backtest_tab():
    st.header("📊 Backtest Results")

    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.selectbox("Symbol", ["MNQ1!", "MES1!"], index=0)
    with col2:
        days = st.selectbox("Period", [30, 90, 180, 365], index=3)

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            start_date = datetime.now() - timedelta(days=days)
            bt_result = st.session_state.signal_engine.run_backtest(
                symbol=symbol,
                start_date=start_date
            )

            if 'error' in bt_result:
                st.error(bt_result['error'])
            else:
                st.session_state.backtest_result = bt_result

    if hasattr(st.session_state, 'backtest_result') and st.session_state.backtest_result:
        bt = st.session_state.backtest_result

        st.divider()
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Total Signals", bt['total_signals'])
        with stat_col2:
            st.metric("Wins", bt['winning_trades'])
        with stat_col3:
            st.metric("Losses", bt['losing_trades'])
        with stat_col4:
            st.metric("Win Rate", f"{bt['win_rate']:.1f}%")

        if bt['signals']:
            signals_df = pd.DataFrame(bt['signals'])
            st.divider()
            st.subheader("Signal History")
            st.dataframe(
                signals_df[['timestamp', 'direction', 'fvg_type', 'session', 'score', 'profit']].tail(20),
                use_container_width=True
            )


def render_research_tab():
    st.header("🔬 Research & Statistics")

    with st.spinner("Loading stats..."):
        fill_stats = st.session_state.signal_engine.get_fvg_fill_stats()

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Signals", fill_stats.get('total', 0))
    with col2:
        st.metric("FVG Fill Rate", f"{fill_stats.get('fill_rate', 0):.1f}%")
    with col3:
        st.metric("Filled", fill_stats.get('filled', 0))

    st.divider()
    st.subheader("Session Performance")
    if fill_stats.get('by_session'):
        session_df = pd.DataFrame(fill_stats['by_session'])
        if not session_df.empty:
            st.dataframe(session_df, use_container_width=True)
    else:
        st.info("No session data available yet")

    st.divider()
    st.subheader("📈 Analysis")

    st.write("""
    **FVG Fill Rate Analysis**
    - Tracks how often price returns to fill the Fair Value Gap
    - Higher fill rates indicate more reliable signals
    
    **Session Breakdown**
    - NY Session: Typically highest volatility
    - London: Good range, liquid
    - Asia: Lower volatility, range-bound
    """)


def main():
    init_session_state()

    st.title("📈 SMT Trading Signal System")
    st.markdown("**MNQ/MES SMT Divergence + FVG Strategy**")

    tab1, tab2, tab3 = st.tabs(["📡 Live", "📊 Backtest", "🔬 Research"])

    with tab1:
        render_live_tab()

    with tab2:
        render_backtest_tab()

    with tab3:
        render_research_tab()

    st.divider()
    st.caption("Data provided via tvdatafeed | Not financial advice")


if __name__ == "__main__":
    main()