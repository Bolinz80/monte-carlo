import os
import sys
import traceback
import random
import numpy as np
import pandas as pd
import matplotlib

# Prefer interactive TkAgg if available, otherwise Agg
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

def prompt_select_file(prompt_title="Select Excel/CSV file"):
    """
    GUI file dialog (tkinter) or CLI fallback — returns selected path or None.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title=prompt_title,
            filetypes=[("Excel files", ("*.xlsx", "*.xls")), ("CSV files", ("*.csv",)), ("All files", ("*.*",))]
        )
        try:
            root.destroy()
        except Exception:
            pass
        return path or None
    except Exception:
        try:
            entry = input(f"{prompt_title} — enter full path (or leave blank to cancel): ").strip()
            return entry if entry else None
        except Exception:
            return None

def estimate_params_from_index_file(path, closing_col_hint='closing_price', trading_days=244):
    """
    Read file (Excel/CSV), detect the closing-price column and return:
    S0 (last price), mu_daily (mean log-return), sigma_daily (std log-return),
    mu_ann (annualized mean), sigma_ann (annualized vol).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(path)
    elif ext in ('.csv',):
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file type: " + ext)

    # Detect closing price column
    col_candidates = {c: str(c).strip().lower() for c in df.columns}
    target = None
    for name, lc in col_candidates.items():
        if lc in {'closing_price', 'close', 'closing', 'closingprice', 'closing price'}:
            target = name
            break
    if target is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 1:
            target = numeric_cols[0]
        elif len(df.columns) == 2:
            for c in df.columns:
                if pd.to_numeric(df[c], errors='coerce').notna().sum() > 0:
                    target = c
                    break
    if target is None:
        raise ValueError("Could not detect closing price column. Provide a file with a numeric closing price column.")

    prices = pd.to_numeric(df[target], errors='coerce').dropna()
    if prices.empty or len(prices) < 2:
        raise ValueError("Not enough price data to estimate returns.")

    # Try to sort by date if a date column exists
    date_cols = [c for c in df.columns if 'date' in str(c).strip().lower()]
    if date_cols:
        try:
            dates = pd.to_datetime(df[date_cols[0]], errors='coerce')
            combined = pd.DataFrame({'date': dates, 'price': pd.to_numeric(df[target], errors='coerce')})
            combined = combined.dropna(subset=['date', 'price']).sort_values('date')
            if not combined.empty:
                prices = combined['price'].astype(float)
        except Exception:
            pass

    logret = np.log(prices / prices.shift(1)).dropna()
    mu_daily = float(logret.mean())
    sigma_daily = float(logret.std(ddof=1))
    mu_ann = mu_daily * trading_days
    sigma_ann = sigma_daily * np.sqrt(trading_days)
    S0 = float(prices.iloc[-1])
    return S0, mu_daily, sigma_daily, mu_ann, sigma_ann

def simulate_gbm(S0, mu_ann, sigma_ann, T=2.0, steps_per_year=244, n_sims=10000, seed=None):
    rng = np.random.default_rng(seed)
    N = int(T * steps_per_year)
    if N <= 0:
        raise ValueError("Non-positive number of steps.")
    dt = 1.0 / steps_per_year
    Z = rng.standard_normal(size=(n_sims, N))
    increments = np.exp((mu_ann - 0.5 * sigma_ann**2) * dt + sigma_ann * np.sqrt(dt) * Z)
    S = np.empty((n_sims, N + 1), dtype=float)
    S[:, 0] = S0
    S[:, 1:] = S0 * np.cumprod(increments, axis=1)
    times = np.linspace(0, T, N + 1)
    return S, times

def calculate_call_option_value(S, strike_price, risk_free_rate, T):
    """
    Calculate the value of a call option using Monte Carlo simulation.
    - S: Simulated paths (n_sims x (steps+1))
    - strike_price: Strike price of the option
    - risk_free_rate: Annualized risk-free rate (e.g., 0.02 for 2%)
    - T: Time to maturity in years
    """
    # Final prices at maturity
    final_prices = S[:, -1]
    # Payoff of the call option
    payoffs = np.maximum(final_prices - strike_price, 0)
    # Discounted expected payoff
    discounted_payoff = np.exp(-risk_free_rate * T) * np.mean(payoffs)
    return discounted_payoff

def plot_paths_and_percentiles(S, times, n_sample=200, out_path="index_montecarlo_paths.png"):
    """
    Plot Monte Carlo paths with left axis for index levels and right axis for return percentages.
    """
    pct = np.percentile(S, [5, 25, 50, 75, 95], axis=0)
    S0 = S[0, 0]  # initial price

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot a subset of paths (n_sample)
    for i in range(min(n_sample, S.shape[0])):
        ax1.plot(times, S[i], color='gray', alpha=0.3, linewidth=0.7)

    # Plot percentiles
    ax1.plot(times, pct[2], color='black', label='Median')
    ax1.fill_between(times, pct[0], pct[4], color='orange', alpha=0.15, label='5-95%')
    ax1.fill_between(times, pct[1], pct[3], color='orange', alpha=0.25, label='25-75%')

    # Left axis: Index level
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Index level')
    ax1.set_title('Monte Carlo GBM paths (2-year horizon)')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Right axis: Return percentages
    ax2 = ax1.twinx()
    ax2.set_ylabel('Return (%)')
    ax2.set_ylim((ax1.get_ylim()[0] / S0 - 1) * 100, (ax1.get_ylim()[1] / S0 - 1) * 100)

    fig.tight_layout()
    try:
        plt.show()
    except Exception:
        pass
    try:
        fig.savefig(out_path)
        print("Saved plot to:", os.path.abspath(out_path))
    except Exception as e:
        print("Failed to save plot:", e)
    plt.close(fig)

if __name__ == "__main__":
    try:
        print("Starting Monte Carlo (index closing) script...")
        # Prompt for file
        data_path = prompt_select_file("Select index_closing Excel/CSV file")
        if not data_path:
            print("No file selected. Exiting.")
            sys.exit(0)

        # Settings
        days_per_year = 244
        T_years = 2.0
        n_sims = 10000  # Number of simulations
        seed = random.randint(0, 1_000_000)  # Generate a random seed
        risk_free_rate = 0.02  # 2% annualized risk-free rate
        strike_price = 100  # Example strike price

        print(f"Using random seed: {seed}")

        # Estimate parameters from file
        S0, mu_daily, sigma_daily, mu_ann, sigma_ann = estimate_params_from_index_file(
            data_path, trading_days=days_per_year
        )

        print(f"Using file: {data_path}")
        print(f"Last closing price (S0): {S0:.4f}")
        print(f"Daily log-mean: {mu_daily:.6f}, Daily log-std: {sigma_daily:.6f}")
        print(f"Annualized mean (mu): {mu_ann:.4f}, Annualized volatility (sigma): {sigma_ann:.4f}")

        # Simulate GBM paths
        S, times = simulate_gbm(S0, mu_ann, sigma_ann, T=T_years, steps_per_year=days_per_year, n_sims=n_sims, seed=seed)

        # Calculate call option value
        call_value = calculate_call_option_value(S, strike_price, risk_free_rate, T_years)
        print(f"Call option value (strike={strike_price}, maturity={T_years} years): {call_value:.4f}")

        # Plot paths
        plot_paths_and_percentiles(S, times, n_sample=200, out_path="index_montecarlo_paths.png")  # Display 200 paths

    except Exception:
        traceback.print_exc()