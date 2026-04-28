"""
Bootstrap confidence interval utility.

WHY: Point estimates are meaningless without uncertainty. Bootstrap resampling
provides non-parametric 95% confidence intervals for any metric.

HOW: Resample the per-example metric values with replacement N times (default
1000), compute the metric on each resample, take the 2.5th and 97.5th percentiles.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    statistic: Callable = np.mean,
    seed: int = 42,
) -> dict[str, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        values: Per-example metric values.
        n_resamples: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        statistic: Function to compute on each resample (default: mean).
        seed: Random seed for reproducibility.

    Returns:
        Dict with point_estimate, ci_lower, ci_upper, ci_width, n.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    point_estimate = float(statistic(values))

    if len(values) < 2:
        return {
            "point_estimate": point_estimate,
            "ci_lower": point_estimate,
            "ci_upper": point_estimate,
            "ci_width": 0.0,
            "n": len(values),
            "n_resamples": 0,
        }

    # Bootstrap resampling
    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        resample = rng.choice(values, size=len(values), replace=True)
        boot_stats[i] = statistic(resample)

    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_stats, 100 * alpha))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha)))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "n": len(values),
        "n_resamples": n_resamples,
    }


def bootstrap_metric(
    results: list[dict[str, Any]],
    metric_fn: Callable[[list[dict[str, Any]]], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap CI for a metric function that operates on per-example result dicts.

    This is for metrics like CBR that need access to full result dicts
    (not just a single scalar per example).

    Args:
        results: List of per-example result dicts.
        metric_fn: Function that takes a list of results and returns a scalar.
        n_resamples: Number of resamples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        Dict with point_estimate, ci_lower, ci_upper, ci_width.
    """
    rng = np.random.default_rng(seed)
    n = len(results)

    point_estimate = float(metric_fn(results))

    if n < 2:
        return {
            "point_estimate": point_estimate,
            "ci_lower": point_estimate,
            "ci_upper": point_estimate,
            "ci_width": 0.0,
            "n": n,
            "n_resamples": 0,
        }

    boot_stats = np.empty(n_resamples)
    indices = np.arange(n)

    for i in range(n_resamples):
        resample_idx = rng.choice(indices, size=n, replace=True)
        resample = [results[j] for j in resample_idx]
        boot_stats[i] = metric_fn(resample)

    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_stats, 100 * alpha))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha)))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "n": n,
        "n_resamples": n_resamples,
    }
