#!/usr/bin/env python
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signals.signal_engine import SignalEngine

if __name__ == "__main__":
    print("Testing Signal Engine...")
    engine = SignalEngine()
    result = engine.analyze("MNQ1!", "CME", "1H", 50)
    print(f"Result: {result.get('score', 'Error: no score')}")
    print(f"Session: {result.get('session', 'N/A')}")
    print(f"Latest FVG: {result.get('latest_fvg', 'None')}")