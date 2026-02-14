"""
Upward Reversal Detection Strategy Variants.

Each strategy returns a boolean Series indicating buy signals.
All strategies look for bullish reversals after a downtrend.
"""

import pandas as pd
import numpy as np
from indicators import macd, rsi, ema, compute_emas, atr
from candlestick import bullish_candle_score


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the dataframe."""
    d = df.copy()
    c = d["close"]

    # MACD
    d["macd_line"], d["macd_signal"], d["macd_hist"] = macd(c)

    # RSI
    d["rsi"] = rsi(c, 14)

    # EMAs
    for p in [9, 21, 50, 200]:
        d[f"ema_{p}"] = ema(c, p)

    # ATR for position sizing / stop-loss
    d["atr"] = atr(d["high"], d["low"], c, 14)

    # Candlestick score
    d["candle_score"] = bullish_candle_score(d["open"], d["high"], d["low"], c)

    # Derived signals
    d["macd_cross_up"] = (d["macd_line"] > d["macd_signal"]) & (
        d["macd_line"].shift(1) <= d["macd_signal"].shift(1)
    )
    d["macd_hist_rising"] = d["macd_hist"] > d["macd_hist"].shift(1)
    d["rsi_oversold_exit"] = (d["rsi"] > 30) & (d["rsi"].shift(1) <= 30)
    d["rsi_bullish_zone"] = (d["rsi"] > 30) & (d["rsi"] < 50)
    d["price_above_ema9"] = c > d["ema_9"]
    d["ema9_above_ema21"] = d["ema_9"] > d["ema_21"]
    d["downtrend"] = c < d["ema_50"]

    return d


# ---------------------------------------------------------------------------
# Strategy Variant 1: Conservative — All signals must align
# ---------------------------------------------------------------------------
def strategy_conservative(df: pd.DataFrame) -> pd.Series:
    """
    Requires ALL of:
    - MACD bullish crossover
    - RSI exiting oversold (crossing above 30)
    - Price above EMA-9
    - At least 1 bullish candlestick pattern
    - Was in downtrend (below EMA-50) recently
    """
    d = prepare_indicators(df)
    recent_downtrend = d["downtrend"].rolling(5).sum() >= 3  # 3 of last 5 bars below EMA-50
    signal = (
        d["macd_cross_up"]
        & d["rsi_oversold_exit"]
        & d["price_above_ema9"]
        & (d["candle_score"] >= 1)
        & recent_downtrend
    )
    return signal


# ---------------------------------------------------------------------------
# Strategy Variant 2: MACD-Focused — MACD + histogram momentum + RSI filter
# ---------------------------------------------------------------------------
def strategy_macd_focused(df: pd.DataFrame) -> pd.Series:
    """
    Requires:
    - MACD bullish crossover
    - MACD histogram rising for 2+ bars
    - RSI in bullish zone (30-50) — not overbought
    - Recent downtrend
    """
    d = prepare_indicators(df)
    hist_rising_2 = d["macd_hist_rising"] & d["macd_hist_rising"].shift(1)
    recent_downtrend = d["downtrend"].rolling(5).sum() >= 3
    signal = (
        d["macd_cross_up"]
        & hist_rising_2
        & d["rsi_bullish_zone"]
        & recent_downtrend
    )
    return signal


# ---------------------------------------------------------------------------
# Strategy Variant 3: RSI-Focused — Deep oversold bounce + confirmation
# ---------------------------------------------------------------------------
def strategy_rsi_focused(df: pd.DataFrame) -> pd.Series:
    """
    Requires:
    - RSI was below 25 within last 3 bars (deep oversold)
    - RSI now crossing above 30
    - MACD histogram turning positive or rising
    - At least 1 bullish candle pattern
    """
    d = prepare_indicators(df)
    deep_oversold = (d["rsi"].shift(1).rolling(3).min() < 25)
    signal = (
        deep_oversold
        & d["rsi_oversold_exit"]
        & (d["macd_hist_rising"] | (d["macd_hist"] > 0))
        & (d["candle_score"] >= 1)
    )
    return signal


# ---------------------------------------------------------------------------
# Strategy Variant 4: EMA Ribbon — EMA alignment + momentum
# ---------------------------------------------------------------------------
def strategy_ema_ribbon(df: pd.DataFrame) -> pd.Series:
    """
    Requires:
    - EMA-9 crosses above EMA-21
    - Price above EMA-9
    - MACD histogram positive
    - RSI between 40-65 (momentum but not overbought)
    """
    d = prepare_indicators(df)
    ema_cross = d["ema9_above_ema21"] & (~d["ema9_above_ema21"].shift(1).fillna(False))
    signal = (
        ema_cross
        & d["price_above_ema9"]
        & (d["macd_hist"] > 0)
        & (d["rsi"] > 40)
        & (d["rsi"] < 65)
    )
    return signal


# ---------------------------------------------------------------------------
# Strategy Variant 5: Candle-First — Candlestick patterns + indicator confirmation
# ---------------------------------------------------------------------------
def strategy_candle_first(df: pd.DataFrame) -> pd.Series:
    """
    Requires:
    - Strong candlestick signal (score >= 2) OR hammer/engulfing with volume-like confirmation
    - MACD histogram rising
    - RSI below 50 (room to run)
    - Recent downtrend context
    """
    d = prepare_indicators(df)
    recent_downtrend = d["downtrend"].rolling(5).sum() >= 2
    signal = (
        (d["candle_score"] >= 2)
        & d["macd_hist_rising"]
        & (d["rsi"] < 50)
        & recent_downtrend
    )
    return signal


# ---------------------------------------------------------------------------
# Strategy Variant 6: Composite Scoring — Weighted signal approach
# ---------------------------------------------------------------------------
def strategy_composite(df: pd.DataFrame, threshold: float = 4.0) -> pd.Series:
    """
    Weighted scoring system:
    - MACD cross up:        +2.0
    - MACD histogram rising: +1.0
    - RSI oversold exit:    +2.0
    - RSI bullish zone:     +1.0
    - EMA-9 > EMA-21:      +1.0
    - Price above EMA-9:    +1.0
    - Candle score:         +1.5 per pattern
    - Recent downtrend:     +1.0 (context requirement)

    Buy when score >= threshold (default 4.0)
    """
    d = prepare_indicators(df)
    recent_downtrend = d["downtrend"].rolling(5).sum() >= 2

    score = pd.Series(0.0, index=d.index)
    score += d["macd_cross_up"].astype(float) * 2.0
    score += d["macd_hist_rising"].astype(float) * 1.0
    score += d["rsi_oversold_exit"].astype(float) * 2.0
    score += d["rsi_bullish_zone"].astype(float) * 1.0
    score += d["ema9_above_ema21"].astype(float) * 1.0
    score += d["price_above_ema9"].astype(float) * 1.0
    score += d["candle_score"] * 1.5
    score += recent_downtrend.astype(float) * 1.0

    return score >= threshold


ALL_STRATEGIES = {
    "conservative": strategy_conservative,
    "macd_focused": strategy_macd_focused,
    "rsi_focused": strategy_rsi_focused,
    "ema_ribbon": strategy_ema_ribbon,
    "candle_first": strategy_candle_first,
    "composite": strategy_composite,
}
