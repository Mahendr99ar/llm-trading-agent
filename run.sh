#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Running LLM Trading Agent (BTC)..."
python main.py --backtest-only
