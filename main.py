import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
# import asyncio


def bs_price(S, K, t, sigma, opt):
    if t <= 0:
        # at expiry, price = intrinsic
        return max(S - K, 0.0) if opt == 'C' else max(K - S, 0.0)
    if sigma <= 0:
        return max(S - K, 0.0) if opt == 'C' else max(K - S, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if opt == 'C':
        return S * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        return K * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, t, opt, lo=1e-6, hi=5.0):
    # Reject impossible prices (below intrinsic or above simple bounds)
    intrinsic = max(S - K, 0.0) if opt == 'C' else max(K - S, 0.0)
    # if price < intrinsic - 1e-12:
    #     return float('nan')
    # For calls with r=0,q=0: upper bound ~ S; for puts: upper bound ~ K
    # if (opt == 'C' and price > S + 1e-12) or (opt == 'P' and price > K + 1e-12):
    #     return float('nan')
    if t <= 0:
        return 0.0 if abs(price - intrinsic) < 1e-12 else float('nan')

    f = lambda sig: bs_price(S, K, t, sig, opt) - price
    f_lo, f_hi = f(lo), f(hi)

    # Ensure a sign change; if not, expand hi a bit
    if f_lo * f_hi > 0:
        for hi_try in (10.0, 20.0, 50.0):
            f_hi = f(hi_try)
            if f_lo * f_hi <= 0:
                hi = hi_try
                break
        else:
            return float('nan')  # couldn't bracket

    try:
        return brentq(f, lo, hi, maxiter=100, xtol=1e-12, rtol=1e-12)
    except Exception:
        return float('nan')


def find_iv_series(df: pd.DataFrame) -> pd.Series:
    """
    df columns required: mark_price (float), dte (float), strike (float), underlying_price (float), option_type ('C' or 'P').
    dte_unit: 'days'.
    Returns a copy with a new 'iv' column (annualized volatility, e.g. 0.55 = 55%).
    """

    t = df["dte"].to_numpy(dtype=float) / 365.0

    price = df["mark_price"].to_numpy(dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    opt = df["option_type"].astype(str).str.upper().to_numpy()
    spot = df["underlying_price"].to_numpy(dtype=float)

    # Compute row-wise (brentq is scalar), loop over NumPy arrays
    ivs = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        ivs[i] = implied_vol(price[i], spot[i], K[i], t[i], 'C' if opt[i] == 'C' else 'P')

    return pd.Series(ivs, index=df.index, name='iv')


def plot_iv_surface(ax, df, title):
    surf = ax.plot_trisurf(
        df["strike"],
        df["dte"],
        df["iv"],
        cmap="viridis",
        linewidth=0.2,
        antialiased=True
    )
    ax.set_title(title)
    ax.set_xlabel("Strike")
    ax.set_ylabel("DTE (days)")
    ax.set_zlabel("IV (%)")
    return surf


def df_from_deribit_orderbooks(orderbooks: list):
    df = pd.DataFrame(orderbooks)

    # Parse expiry/strike/type from instrument_name like "ETH-26JUN26-4500-C"
    parts = df["instrument_name"].str.split("-", expand=True)
    df["expiry_str"]   = parts[1]
    df["strike"]       = parts[2].astype(float)
    df["option_type"]  = parts[3]

    # Expiry as a real datetime (DDMMMYY → 2026-06-26)
    df["expiry"] = pd.to_datetime(df["expiry_str"], format="%d%b%y", errors="coerce")

    # might want to pass now variable as an argument defined at the time of the request
    df["now"] = pd.Timestamp.utcnow()

    dte_seconds = (df["expiry"].dt.tz_localize("UTC") - df["now"]).dt.total_seconds()
    df["dte"] = dte_seconds / 86400.0

    df["iv"] = find_iv_series(df)      

    return df[["strike", "iv", "expiry", "dte", "option_type", "instrument_name"]].sort_values(["expiry","strike"])



def get_deribit_book_summaries(currency: str = "ETH"):
    book_summaries_url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={currency}&kind=option"
    try:
        resp = requests.get(book_summaries_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise ValueError(f"Deribit API error: {data['error']}")

        active_orderbooks = []
        orderbooks = data.get("result", [])
        for orderbook in orderbooks:
            if "mark_iv" in orderbook and orderbook["mark_iv"] is not None:
                active_orderbooks.append(orderbook)
        return active_orderbooks
        
    except requests.RequestException as e:
        print(f"Request failed: {e}")
    except ValueError as e:
        print(f"Data error: {e}")

    return []


def main():
    deribit_orderbooks = get_deribit_book_summaries()
    deribit_df = df_from_deribit_orderbooks(deribit_orderbooks)
    deribit_calls = deribit_df[deribit_df["option_type"] == "C"]
    print(deribit_calls.head())
    deribit_puts  = deribit_df[deribit_df["option_type"] == "P"]
    print(deribit_puts.sort_values("strike", ascending=False).head())

    fig1 = plt.figure(figsize=(9, 6))

    ax1 = fig1.add_subplot(121, projection="3d")
    surf1 = plot_iv_surface(ax1, deribit_calls, "Deribit - Calls")
    fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label="IV (%)")

    ax2 = fig1.add_subplot(122, projection="3d")
    surf2 = plot_iv_surface(ax2, deribit_puts, "Deribit - Puts")
    fig1.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label="IV (%)")

    fig2 = plt.figure(figsize=(6, 6))
    ax3 = fig2.add_subplot(111, projection="3d")
    surf3 = plot_iv_surface(ax3, deribit_df, "Deribit - All Options")
    fig2.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, label="IV (%)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()



