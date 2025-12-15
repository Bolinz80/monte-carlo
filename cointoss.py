import numpy as np

# ---- Matplotlib backend safety ----
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt

from typing import Optional, Tuple


def simulate_coin_walk(N: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    steps = rng.choice([1, -1], size=N)
    cumulative = np.cumsum(steps)
    running_mean = cumulative / np.arange(1, N + 1)
    return cumulative, running_mean


def simulate_ensemble_mean_path(paths: int, N: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.choice([1, -1], size=(paths, N))
    cumulative = np.cumsum(steps, axis=1)
    return cumulative.mean(axis=0)


def simulate_many_final_sums(paths: int, N: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.choice([1, -1], size=(paths, N))
    return steps.sum(axis=1)


def main() -> None:
    N = 10_000
    paths_mean = 1_000
    paths_hist = 100_000
    seed: Optional[int] = 42

    # Simulations
    cum, rmean = simulate_coin_walk(N, seed)
    avg_cum = simulate_ensemble_mean_path(paths_mean, N, seed)
    final_sums = simulate_many_final_sums(paths_hist, N, seed)

    # ---- One figure, four panels ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Running mean
    ax = axes[0, 0]
    ax.plot(rmean)
    ax.axhline(0)
    ax.set_title("Running Mean → 0")
    ax.set_xlabel("Toss")
    ax.set_ylabel("Average")

    # (2) Single-path cumulative sum
    ax = axes[0, 1]
    ax.plot(cum)
    ax.axhline(0)
    ax.set_title("Cumulative Sum (Noise Accumulation ~ √N)")
    ax.set_xlabel("Toss")
    ax.set_ylabel("Sum")

    # (3) Ensemble mean of cumulative sum
    ax = axes[1, 0]
    ax.plot(avg_cum)
    ax.axhline(0)
    ax.set_title(f"Ensemble Mean Path ({paths_mean} paths)")
    ax.set_xlabel("Toss")
    ax.set_ylabel("Average Sum")

    # (4) Distribution of final sums
    ax = axes[1, 1]
    ax.hist(final_sums, bins=60, density=True)
    ax.set_title("Distribution of Final Sum (CLT)")
    ax.set_xlabel("Final Sum")
    ax.set_ylabel("Density")

    plt.suptitle("Coin Toss Monte Carlo: Mean vs Sum vs Distribution", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
