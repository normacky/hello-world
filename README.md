# Stock Upward Reversal Detection Strategy

Detects bullish reversals using MACD, RSI, EMAs, and candlestick pattern analysis. Includes 6 strategy variants and a backtesting engine to compare them.

## Strategy Variants

| # | Strategy | Description |
|---|----------|-------------|
| 1 | **Conservative** | Requires ALL signals: MACD cross + RSI oversold exit + price > EMA-9 + candlestick pattern + downtrend context |
| 2 | **MACD-Focused** | MACD bullish crossover + rising histogram (2+ bars) + RSI in bullish zone (30-50) |
| 3 | **RSI-Focused** | Deep oversold bounce (RSI < 25) + crossing above 30 + MACD confirmation + candlestick pattern |
| 4 | **EMA Ribbon** | EMA-9 crossing above EMA-21 + price > EMA-9 + positive MACD histogram + RSI 40-65 |
| 5 | **Candle-First** | Strong candlestick score (2+) + rising MACD histogram + RSI < 50 + downtrend context |
| 6 | **Composite** | Weighted scoring across all indicators; fires when combined score >= threshold |

## Backtest Results (Best Strategy: Composite)

Tested across 5 market scenarios (trending up, high volatility, choppy sideways, strong downtrend, moderate cycles), 500 bars each.

| Strategy | Total Trades | Total PnL% | Avg Win Rate | Profit Factor | Avg Sharpe |
|----------|-------------|------------|--------------|---------------|------------|
| **Composite** | **172** | **+4.35%** | **48.2%** | **1.07** | **0.02** |
| EMA Ribbon | 101 | -29.80% | 48.7% | 0.97 | -0.03 |
| MACD-Focused | 30 | -22.21% | 46.7% | 0.83 | -0.29 |

The **composite scoring strategy** won because it adapts to different market conditions by weighting multiple indicators rather than requiring rigid boolean conditions.

## Indicators Used

- **MACD** (12, 26, 9) — trend momentum and crossover signals
- **RSI** (14) — oversold/overbought detection
- **EMAs** (9, 21, 50, 200) — trend direction and support levels
- **ATR** (14) — volatility-based stop-loss and take-profit
- **Candlestick patterns** — hammer, bullish engulfing, morning star, dragonfly doji

## Files

- `indicators.py` — Technical indicator calculations
- `candlestick.py` — Candlestick pattern detection
- `strategies.py` — 6 reversal detection strategy variants
- `backtest.py` — Backtesting engine with synthetic data generation

## Usage

```bash
pip install pandas numpy matplotlib
python backtest.py
```
