"""
BACKTESTING ENGINE
- Simulates trading based on signals
- Calculates Sharpe ratio, max drawdown, win rate
- Like WorldQuant BRAIN's backtesting but custom built
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    avg_trade_return: float
    final_capital: float


# ─────────────────────────────────────────
# BACKTESTING ENGINE
# ─────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100_000,
    position_size: float = 0.95,    # 95% of capital per trade
    stop_loss: float = 0.05,        # 5% stop loss
    take_profit: float = 0.10,      # 10% take profit
    transaction_cost: float = 0.001 # 0.1% per trade
) -> tuple[BacktestResult, pd.DataFrame]:
    """
    Simulate trading on historical signals.
    
    Returns:
        BacktestResult: performance metrics
        pd.DataFrame: trade-by-trade details
    """
    capital   = initial_capital
    position  = 0.0      # current shares held
    entry_px  = 0.0      # entry price
    trades    = []
    equity    = []

    for i, row in df.iterrows():
        price  = row["close"]
        signal = row.get("signal", "HOLD")

        # ── ENTRY ──
        if signal == "BUY" and position == 0:
            shares    = (capital * position_size) / price
            cost      = shares * price * (1 + transaction_cost)
            if cost <= capital:
                capital  -= cost
                position  = shares
                entry_px  = price

        # ── EXIT ──
        elif position > 0:
            pnl_pct = (price - entry_px) / entry_px

            should_exit = (
                signal == "SELL"
                or pnl_pct <= -stop_loss    # stop loss hit
                or pnl_pct >= take_profit   # take profit hit
            )

            if should_exit:
                proceeds = position * price * (1 - transaction_cost)
                capital += proceeds
                trade_return = (price - entry_px) / entry_px

                trades.append({
                    "date":         str(i.date()),
                    "entry_price":  entry_px,
                    "exit_price":   price,
                    "return_pct":   round(trade_return * 100, 2),
                    "profitable":   trade_return > 0,
                    "exit_reason":  signal if signal == "SELL" else
                                    ("STOP_LOSS" if pnl_pct <= -stop_loss else "TAKE_PROFIT"),
                })
                position = 0
                entry_px = 0

        # Total equity = cash + current position value
        total_equity = capital + (position * price)
        equity.append({"date": str(i.date()), "equity": total_equity})

    # Close any open position at end
    if position > 0:
        final_price = df["close"].iloc[-1]
        capital    += position * final_price * (1 - transaction_cost)

    # ── METRICS ──
    equity_df = pd.DataFrame(equity)
    equity_df["returns"] = equity_df["equity"].pct_change().fillna(0)

    # Sharpe ratio (annualized)
    mean_ret = equity_df["returns"].mean()
    std_ret  = equity_df["returns"].std() + 1e-9
    sharpe   = (mean_ret / std_ret) * np.sqrt(252)

    # Max Drawdown
    peak     = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - peak) / peak
    max_dd   = drawdown.min()

    # Win rate
    total_trades      = len(trades)
    profitable_trades = sum(1 for t in trades if t["profitable"])
    win_rate          = profitable_trades / max(total_trades, 1)
    avg_trade_return  = np.mean([t["return_pct"] for t in trades]) if trades else 0

    result = BacktestResult(
        total_return     = round((capital - initial_capital) / initial_capital * 100, 2),
        sharpe_ratio     = round(float(sharpe), 3),
        max_drawdown     = round(float(max_dd) * 100, 2),
        win_rate         = round(win_rate * 100, 2),
        total_trades     = total_trades,
        profitable_trades= profitable_trades,
        avg_trade_return = round(avg_trade_return, 2),
        final_capital    = round(capital, 2),
    )

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return result, trades_df


# ─────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────

def print_results(result: BacktestResult, symbol: str = ""):
    print("\n" + "="*50)
    print(f"  BACKTEST RESULTS  {symbol}")
    print("="*50)
    print(f"  Total Return    : {result.total_return:+.2f}%")
    print(f"  Sharpe Ratio    : {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown    : {result.max_drawdown:.2f}%")
    print(f"  Win Rate        : {result.win_rate:.1f}%")
    print(f"  Total Trades    : {result.total_trades}")
    print(f"  Profitable      : {result.profitable_trades}")
    print(f"  Avg Trade Return: {result.avg_trade_return:+.2f}%")
    print(f"  Final Capital   : ₹{result.final_capital:,.0f}")
    print("="*50 + "\n")
