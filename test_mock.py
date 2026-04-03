#!/usr/bin/env python
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signals.signal_engine import SignalEngine
from src.signals.fvg_detector import FVGDetector

def create_mock_data(symbol="MNQ1!", bars=100):
    dates = [datetime.now() - timedelta(hours=bars-i) for i in range(bars)]
    np.random.seed(42)
    data = {
        'open': 18000 + np.random.randn(bars).cumsum(),
        'high': 18050 + np.random.randn(bars).cumsum(),
        'low': 17950 + np.random.randn(bars).cumsum(),
        'close': 18000 + np.random.randn(bars).cumsum(),
        'volume': np.random.randint(1000, 10000, bars)
    }
    df = pd.DataFrame(data, index=dates)
    return df

def test_fvg_detector():
    print("Testing FVG Detector...")
    df = create_mock_data()
    detector = FVGDetector()
    result = detector.detect(df)
    fvgs = detector.get_latest(result, n=5)
    print(f"  Found {len(fvgs)} FVGs in mock data")
    print(f"  Latest FVG: {fvgs[-1] if fvgs else 'None'}")
    return result

def test_signal_engine():
    print("\nTesting Signal Engine (with mock data)...")
    engine = SignalEngine()
    
    original_get_data = engine.data_manager.get_data
    def mock_get_data(symbol, exchange="CME", interval="1H", n_bars=200):
        print(f"  [MOCK] Returning mock data for {symbol}")
        return create_mock_data(symbol, n_bars)
    
    engine.data_manager.get_data = mock_get_data
    
    result = engine.analyze("MNQ1!", "CME", "1H", 50)
    print(f"  Score: {result.get('score', 'N/A')}")
    print(f"  Grade: {result.get('grade', 'N/A')}")
    print(f"  Session: {result.get('session', 'N/A')}")
    print(f"  Latest FVG: {result.get('latest_fvg', 'None')}")
    print(f"  HTF 4H Bias: {result.get('htf_4h_bias', 'N/A')}")
    print(f"  HTF Daily Bias: {result.get('htf_daily_bias', 'N/A')}")
    print(f"  Liquidity Sweep: {result.get('liquidity_sweep', 'N/A')}")
    print(f"  OB Overlap: {result.get('ob_overlap', 'N/A')}")
    print(f"  SMT Divergence: {result.get('smt_divergence', 'N/A')}")
    return result

if __name__ == "__main__":
    print("=" * 50)
    print("SMT Trading Signal System - Mock Data Test")
    print("=" * 50)
    
    test_fvg_detector()
    test_signal_engine()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)