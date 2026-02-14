"""
Backtesting engine for reversal detection strategies.

Uses synthetic market data with realistic characteristics (trends, mean reversion,
volatility clustering) to test each strategy variant across multiple market scenarios.
"""

import numpy as np
import pandas as pd
from strategies import ALL_STRATEGIES, prepare_indicators


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def generate_market_data(
    n_days: int = 500,
    seed: int = 42,
    initial_price: float = 100.0,
    trend_strength: float = 0.0002,
    volatility: float = 0.02,
    mean_reversion: float = 0.01,
    n_reversals: int = 5,
) -> pd.DataFrame:
    """
    Generate realistic OHLC data with embedded reversals.
    Includes regime changes (uptrend -> downtrend -> reversal) to test strategies.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2023-01-01", periods=n_days)

    prices = np.zeros(n_days)
    prices[0] = initial_price

    # Create regime schedule: alternating trends with reversals
    regime_length = n_days // (n_reversals * 2 + 1)
    regimes = []
    for i in range(n_reversals * 2 + 1):
        if i % 2 == 0:
            regimes.extend(["up"] * regime_length)
        else:
            regimes.extend(["down"] * regime_length)
    regimes = regimes[:n_days]
    while len(regimes) < n_days:
        regimes.append("up")

    for i in range(1, n_days):
        vol = volatility * (1 + 0.5 * np.sin(i / 50))  # volatility clustering
        if regimes[i] == "up":
            drift = abs(trend_strength)
        else:
            drift = -abs(trend_strength) * 1.5  # downtrends tend to be sharper

        noise = rng.normal(0, vol)
        ret = drift + noise
        prices[i] = prices[i - 1] * (1 + ret)

    # Generate OHLC from close prices
    opens = np.zeros(n_days)
    highs = np.zeros(n_days)
    lows = np.zeros(n_days)

    opens[0] = prices[0]
    for i in range(1, n_days):
        opens[i] = prices[i - 1] + rng.normal(0, volatility * prices[i - 1] * 0.3)
        intraday_range = abs(rng.normal(0, volatility * prices[i] * 0.8))
        highs[i] = max(opens[i], prices[i]) + abs(rng.normal(0, intraday_range * 0.5))
        lows[i] = min(opens[i], prices[i]) - abs(rng.normal(0, intraday_range * 0.5))

    highs[0] = prices[0] * 1.01
    lows[0] = prices[0] * 0.99

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------
def backtest_strategy(
    df: pd.DataFrame,
    signals: pd.Series,
    holding_period: int = 10,
    stop_loss_atr_mult: float = 2.0,
    take_profit_atr_mult: float = 3.0,
) -> dict:
    """
    Backtest a strategy with fixed holding period and ATR-based stops.

    Returns dict with performance metrics.
    """
    d = prepare_indicators(df)
    atr_values = d["atr"]

    trades = []
    in_trade = False
    entry_bar = 0

    for i in range(len(df)):
        if in_trade:
            bars_held = i - entry_bar
            current_price = df["close"].iloc[i]
            pnl_pct = (current_price - entry_price) / entry_price * 100

            # Check stop loss
            if current_price <= stop_price:
                trades.append({
                    "entry_date": df.index[entry_bar],
                    "exit_date": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": stop_price,
                    "pnl_pct": (stop_price - entry_price) / entry_price * 100,
                    "bars_held": bars_held,
                    "exit_reason": "stop_loss",
                })
                in_trade = False
                continue

            # Check take profit
            if current_price >= take_profit_price:
                trades.append({
                    "entry_date": df.index[entry_bar],
                    "exit_date": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": take_profit_price,
                    "pnl_pct": (take_profit_price - entry_price) / entry_price * 100,
                    "bars_held": bars_held,
                    "exit_reason": "take_profit",
                })
                in_trade = False
                continue

            # Check holding period expiry
            if bars_held >= holding_period:
                trades.append({
                    "entry_date": df.index[entry_bar],
                    "exit_date": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "bars_held": bars_held,
                    "exit_reason": "time_exit",
                })
                in_trade = False
                continue

        elif signals.iloc[i]:
            # Enter trade
            in_trade = True
            entry_bar = i
            entry_price = df["close"].iloc[i]
            current_atr = atr_values.iloc[i] if not np.isnan(atr_values.iloc[i]) else entry_price * 0.02
            stop_price = entry_price - stop_loss_atr_mult * current_atr
            take_profit_price = entry_price + take_profit_atr_mult * current_atr

    return compute_metrics(trades)


def compute_metrics(trades: list[dict]) -> dict:
    """Compute performance metrics from a list of trades."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "total_pnl_pct": 0.0,
            "max_win_pct": 0.0,
            "max_loss_pct": 0.0,
            "profit_factor": 0.0,
            "avg_bars_held": 0.0,
            "sharpe_approx": 0.0,
            "stop_loss_exits": 0,
            "take_profit_exits": 0,
            "time_exits": 0,
        }

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]

    gross_profit = wins["pnl_pct"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl_pct"].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    pnl_std = tdf["pnl_pct"].std()
    sharpe = tdf["pnl_pct"].mean() / pnl_std if pnl_std > 0 else 0

    return {
        "total_trades": len(tdf),
        "win_rate": len(wins) / len(tdf) * 100,
        "avg_pnl_pct": tdf["pnl_pct"].mean(),
        "total_pnl_pct": tdf["pnl_pct"].sum(),
        "max_win_pct": tdf["pnl_pct"].max(),
        "max_loss_pct": tdf["pnl_pct"].min(),
        "profit_factor": profit_factor,
        "avg_bars_held": tdf["bars_held"].mean(),
        "sharpe_approx": sharpe,
        "stop_loss_exits": (tdf["exit_reason"] == "stop_loss").sum(),
        "take_profit_exits": (tdf["exit_reason"] == "take_profit").sum(),
        "time_exits": (tdf["exit_reason"] == "time_exit").sum(),
    }


# ---------------------------------------------------------------------------
# Run backtests across multiple market scenarios
# ---------------------------------------------------------------------------
def run_full_backtest():
    """Run all strategies across multiple market scenarios and report results."""

    scenarios = {
        "trending_up": {"seed": 42, "trend_strength": 0.0005, "volatility": 0.015, "n_reversals": 4},
        "high_volatility": {"seed": 123, "trend_strength": 0.0002, "volatility": 0.035, "n_reversals": 6},
        "choppy_sideways": {"seed": 456, "trend_strength": 0.00005, "volatility": 0.02, "n_reversals": 8},
        "strong_downtrend": {"seed": 789, "trend_strength": 0.0008, "volatility": 0.025, "n_reversals": 3},
        "moderate_cycles": {"seed": 101, "trend_strength": 0.0003, "volatility": 0.02, "n_reversals": 5},
    }

    all_results = {}

    for scenario_name, params in scenarios.items():
        df = generate_market_data(n_days=500, **params)
        scenario_results = {}

        for strat_name, strat_func in ALL_STRATEGIES.items():
            signals = strat_func(df)
            metrics = backtest_strategy(df, signals)
            scenario_results[strat_name] = metrics

        all_results[scenario_name] = scenario_results

    return all_results


def print_results(all_results: dict):
    """Print formatted backtest results."""

    # Aggregate metrics across scenarios
    aggregated = {}
    for strat_name in ALL_STRATEGIES:
        agg = {
            "total_trades": 0,
            "total_wins": 0,
            "total_pnl": 0.0,
            "profit_factors": [],
            "sharpes": [],
            "win_rates": [],
        }
        for scenario_name, scenario_results in all_results.items():
            m = scenario_results[strat_name]
            agg["total_trades"] += m["total_trades"]
            agg["total_pnl"] += m["total_pnl_pct"]
            if m["total_trades"] > 0:
                agg["profit_factors"].append(m["profit_factor"])
                agg["sharpes"].append(m["sharpe_approx"])
                agg["win_rates"].append(m["win_rate"])
        aggregated[strat_name] = agg

    # Print per-scenario details
    print("=" * 100)
    print("BACKTEST RESULTS — UPWARD REVERSAL DETECTION STRATEGIES")
    print("=" * 100)

    for scenario_name, scenario_results in all_results.items():
        print(f"\n{'─' * 100}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'─' * 100}")
        print(
            f"{'Strategy':<18} {'Trades':>7} {'WinRate':>8} {'AvgPnL%':>9} "
            f"{'TotalPnL%':>10} {'ProfitF':>8} {'Sharpe':>8} {'SL/TP/Time':>12}"
        )
        print("-" * 90)

        for strat_name in ALL_STRATEGIES:
            m = scenario_results[strat_name]
            sl_tp_time = f"{m['stop_loss_exits']}/{m['take_profit_exits']}/{m['time_exits']}"
            pf = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "inf"
            print(
                f"{strat_name:<18} {m['total_trades']:>7} {m['win_rate']:>7.1f}% "
                f"{m['avg_pnl_pct']:>8.2f}% {m['total_pnl_pct']:>9.2f}% "
                f"{pf:>8} {m['sharpe_approx']:>7.2f} {sl_tp_time:>12}"
            )

    # Print aggregated ranking
    print(f"\n{'=' * 100}")
    print("AGGREGATED RANKING (across all scenarios)")
    print(f"{'=' * 100}")
    print(
        f"{'Rank':<6} {'Strategy':<18} {'TotalTrades':>12} {'TotalPnL%':>10} "
        f"{'AvgWinRate':>10} {'AvgPF':>8} {'AvgSharpe':>10}"
    )
    print("-" * 80)

    # Rank by composite score: weighted sum of normalized metrics
    # Strategies with 0 trades are ranked last
    ranking = []
    for strat_name, agg in aggregated.items():
        avg_wr = np.mean(agg["win_rates"]) if agg["win_rates"] else 0
        pf_values = [pf for pf in agg["profit_factors"] if pf != float("inf") and not np.isnan(pf)]
        avg_pf = np.mean(pf_values) if pf_values else 0
        sharpe_values = [s for s in agg["sharpes"] if not np.isnan(s)]
        avg_sharpe = np.mean(sharpe_values) if sharpe_values else 0

        if agg["total_trades"] < 10:
            composite = -1.0  # too few trades = unreliable, rank last
        else:
            avg_pnl_per_trade = agg["total_pnl"] / agg["total_trades"]
            # Composite score: emphasize profit factor and sharpe
            composite = (
                avg_wr / 100 * 0.20
                + min(avg_pf, 5) / 5 * 0.30
                + max(min(avg_sharpe, 2), -2) / 2 * 0.30  # clamp sharpe
                + max(min(avg_pnl_per_trade, 5), -5) / 5 * 0.20
            )

        ranking.append({
            "name": strat_name,
            "total_trades": agg["total_trades"],
            "total_pnl": agg["total_pnl"],
            "avg_wr": avg_wr,
            "avg_pf": avg_pf,
            "avg_sharpe": avg_sharpe,
            "composite": composite,
        })

    ranking.sort(key=lambda x: x["composite"], reverse=True)

    for rank, r in enumerate(ranking, 1):
        pf_str = f"{r['avg_pf']:.2f}"
        print(
            f"{rank:<6} {r['name']:<18} {r['total_trades']:>12} {r['total_pnl']:>9.2f}% "
            f"{r['avg_wr']:>9.1f}% {pf_str:>8} {r['avg_sharpe']:>9.2f}"
        )

    # Announce winner
    winner = ranking[0]
    print(f"\n{'*' * 80}")
    print(f"  BEST STRATEGY: {winner['name'].upper()}")
    print(f"  Composite Score: {winner['composite']:.4f}")
    print(f"  Avg Win Rate: {winner['avg_wr']:.1f}%  |  Avg Profit Factor: {winner['avg_pf']:.2f}")
    print(f"  Avg Sharpe: {winner['avg_sharpe']:.2f}  |  Total PnL: {winner['total_pnl']:.2f}%")
    print(f"{'*' * 80}")

    return ranking


if __name__ == "__main__":
    results = run_full_backtest()
    ranking = print_results(results)
