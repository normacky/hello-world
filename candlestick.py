"""Candlestick pattern detection for reversal signals."""

import pandas as pd
import numpy as np


def body_size(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Absolute candle body size."""
    return (close - open_).abs()


def upper_shadow(high: pd.Series, open_: pd.Series, close: pd.Series) -> pd.Series:
    """Upper shadow length."""
    return high - pd.concat([open_, close], axis=1).max(axis=1)


def lower_shadow(low: pd.Series, open_: pd.Series, close: pd.Series) -> pd.Series:
    """Lower shadow length."""
    return pd.concat([open_, close], axis=1).min(axis=1) - low


def candle_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """Total candle range (high - low)."""
    return high - low


def is_hammer(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
              body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> pd.Series:
    """
    Hammer pattern: small body at top, long lower shadow.
    Bullish reversal signal at bottom of downtrend.
    """
    body = body_size(open_, close)
    rng = candle_range(high, low)
    ls = lower_shadow(low, open_, close)
    us = upper_shadow(high, open_, close)

    small_body = body <= rng * body_ratio
    long_lower = ls >= body * shadow_ratio
    short_upper = us <= body * 0.5
    nonzero = rng > 0

    return small_body & long_lower & short_upper & nonzero


def is_bullish_engulfing(open_: pd.Series, close: pd.Series) -> pd.Series:
    """
    Bullish engulfing: previous red candle fully engulfed by current green candle.
    """
    prev_red = close.shift(1) < open_.shift(1)
    curr_green = close > open_
    engulfs_body = (open_ <= close.shift(1)) & (close >= open_.shift(1))
    return prev_red & curr_green & engulfs_body


def is_morning_star(open_: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series) -> pd.Series:
    """
    Morning star: 3-candle pattern.
    1) Large red candle
    2) Small body candle (gap down)
    3) Large green candle closing above midpoint of candle 1
    """
    body = body_size(open_, close)
    avg_body = body.rolling(20).mean()

    # Candle 1: large red
    large_red = (close.shift(2) < open_.shift(2)) & (body.shift(2) > avg_body.shift(2))
    # Candle 2: small body
    small_mid = body.shift(1) < avg_body.shift(1) * 0.5
    # Candle 3: large green closing above midpoint of candle 1
    midpoint = (open_.shift(2) + close.shift(2)) / 2
    large_green = (close > open_) & (close > midpoint) & (body > avg_body * 0.5)

    return large_red & small_mid & large_green


def is_dragonfly_doji(open_: pd.Series, high: pd.Series, low: pd.Series,
                      close: pd.Series) -> pd.Series:
    """
    Dragonfly doji: open and close near high, long lower shadow.
    Bullish reversal at bottom.
    """
    body = body_size(open_, close)
    rng = candle_range(high, low)
    ls = lower_shadow(low, open_, close)

    tiny_body = body <= rng * 0.1
    long_lower = ls >= rng * 0.6
    nonzero = rng > 0

    return tiny_body & long_lower & nonzero


def bullish_candle_score(open_: pd.Series, high: pd.Series, low: pd.Series,
                         close: pd.Series) -> pd.Series:
    """
    Combined bullish candlestick score (0-4).
    Each detected pattern adds 1 point.
    """
    score = pd.Series(0, index=open_.index, dtype=float)
    score += is_hammer(open_, high, low, close).astype(float)
    score += is_bullish_engulfing(open_, close).astype(float)
    score += is_morning_star(open_, high, low, close).astype(float)
    score += is_dragonfly_doji(open_, high, low, close).astype(float)
    return score
