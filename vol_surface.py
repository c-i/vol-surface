import logging
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, newton
from datetime import datetime, timezone
import argparse
import py_vollib.black_scholes.implied_volatility as vollib_iv
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


def iv_brentq(price, S, K, t, opt, lo=1e-6, hi=5.0):
    # Reject impossible prices (below intrinsic or above simple bounds)
    intrinsic = max(S - K, 0.0) if opt == 'C' else max(K - S, 0.0)
    if price < intrinsic - 1e-12:
        return float('nan')
    # For calls with r=0,q=0: upper bound ~ S; for puts: upper bound ~ K
    if (opt == 'C' and price > S + 1e-12) or (opt == 'P' and price > K + 1e-12):
        return float('nan')
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
            logging.warning(f"brentq failed to bracket: {lo}, {hi}")
            return float('nan')  # couldn't bracket

    try:
        return brentq(f, lo, hi, maxiter=100, xtol=1e-12, rtol=1e-12)
    except Exception as e:
        logging.error(f"brentq failed: {e}")
        return float('nan')


# def iv_newton(price, S, K, t, opt, x0):
#     if t <= 0:
#         return 0.0 if abs(price - intrinsic) < 1e-12 else float('nan')

#     f = lambda sig: bs_price(S, K, t, sig, opt) - price
#     try:
#         return newton(f, x0, maxiter=100, tol=1e-12)
#     except Exception as e:
#         logging.error(f"newton failed: {e}")
#         return float('nan')


def iv_vollib(price, S, K, t, opt):
    # Reject impossible prices (below intrinsic or above simple bounds)
    intrinsic = max(S - K, 0.0) if opt == 'C' else max(K - S, 0.0)
    if price < intrinsic - 1e-12:
        return float('nan')
    # For calls with r=0,q=0: upper bound ~ S; for puts: upper bound ~ K
    if (opt == 'C' and price > S + 1e-12) or (opt == 'P' and price > K + 1e-12):
        return float('nan')
    if t <= 0:
        return 0.0 if abs(price - intrinsic) < 1e-12 else float('nan')

    return vollib_iv.implied_volatility(
        price=price,
        S=S,
        K=K,
        t=t,
        r=0.0,
        flag=opt.lower()
    )


def find_iv_series(df: pd.DataFrame, price_type: str, use_vollib: bool) -> pd.Series:
    """
    df columns required: mark_price (float), dte (float), strike (float), underlying_price (float), option_type ('C' or 'P').
    dte_unit: 'days'.
    Returns a copy with a new 'iv' column (annualized volatility, e.g. 0.55 = 55%).
    """
    if price_type == "mark":
        price = df["mark_price"].to_numpy(dtype=float)
    elif price_type == "mid":
        price = df["mid_price"].to_numpy(dtype=float)
    elif price_type == "last":
        price = df["last"].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown price_type: {price_type}")

    t = df["dte"].to_numpy(dtype=float) / 365.0
    K = df["strike"].to_numpy(dtype=float)
    opt = df["option_type"].to_numpy(dtype=str)
    spot = df["underlying_price"].to_numpy(dtype=float)

    # Compute row-wise (brentq is scalar), loop over NumPy arrays
    ivs = np.empty(len(df), dtype=float)
    if use_vollib:
        for i in range(len(df)):
            ivs[i] = iv_vollib(price[i], spot[i], K[i], t[i], opt[i])
    else:
        for i in range(len(df)):
            ivs[i] = iv_brentq(price[i], spot[i], K[i], t[i], opt[i])

    return pd.Series(ivs, index=df.index, name='iv')


def df_from_deribit_orderbooks(orderbooks: list, use_vollib: bool):
    df = pd.DataFrame(orderbooks)

    df["mid_price"] = ((df["bid_price"] + df["ask_price"]) / 2).where(
        df["bid_price"].notna() & df["ask_price"].notna(),
        df["mark_price"]
    )

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

    df["iv"] = find_iv_series(df, "mark", use_vollib)      

    return df[["strike", "iv", "expiry", "dte", "option_type", "instrument_name", "underlying_price"]].sort_values(["expiry","strike"])



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
    ax.set_zlabel("IV")
    return surf


def plot_iv_scatter(ax, df, title):
    scatter = ax.scatter(
        df["strike"],
        df["dte"],
        df["iv"],
        s=8
    )
    ax.set_title(title)
    ax.set_xlabel("Strike")
    ax.set_ylabel("DTE (days)")
    ax.set_zlabel("IV")
    return scatter


def get_args():
    parser = argparse.ArgumentParser(description="Deribit IV Surface Plot")
    parser.add_argument("currencies", type=str, nargs="+", help="Currencies to plot separated by spaces. e.g. BTC ETH")
    parser.add_argument("--scatter", action="store_true", help="Use scatter plot instead of surface plot")
    parser.add_argument("--jaeckel", action="store_true", help="Use py_vollib (Jaeckel's method) to calculate implied volatility")
    return parser.parse_args()


def main():
    args = get_args()
    currencies = args.currencies
    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    for currency in currencies:
        deribit_orderbooks = get_deribit_book_summaries(currency)
        deribit_df = df_from_deribit_orderbooks(deribit_orderbooks, args.jaeckel)
        deribit_df = deribit_df.loc[(deribit_df["iv"] != 0) & (~deribit_df["iv"].isna())]

        spot_price = deribit_df["underlying_price"].iloc[0]
        deribit_otm = deribit_df[
            ((deribit_df["option_type"] == "C") & (deribit_df["strike"] > spot_price)) |
            ((deribit_df["option_type"] == "P") & (deribit_df["strike"] < spot_price))
        ]
        deribit_calls = deribit_df[deribit_df["option_type"] == "C"]
        print(deribit_calls.head())
        deribit_puts  = deribit_df[deribit_df["option_type"] == "P"]
        print(deribit_puts.sort_values("iv").head())

        fig1 = plt.figure(figsize=(9, 6))
        ax1 = fig1.add_subplot(121, projection="3d")
        ax2 = fig1.add_subplot(122, projection="3d")

        fig2 = plt.figure(figsize=(9, 9))
        ax3 = fig2.add_subplot(111, projection="3d")
        
        if args.scatter:
            surf1 = plot_iv_scatter(ax1, deribit_calls, f"Deribit - {currency} - Calls\nTime (UTC): {utc_now} - Spot: {spot_price}")
            surf2 = plot_iv_scatter(ax2, deribit_puts, f"Deribit - {currency} - Puts")
            surf3 = plot_iv_scatter(ax3, deribit_otm, f"Deribit - {currency} - OTM Puts and Calls\nTime (UTC): {utc_now} - Spot: {spot_price}")
        else:
            surf1 = plot_iv_surface(ax1, deribit_calls, f"Deribit - {currency} - Calls\nTime (UTC): {utc_now} - Spot: {spot_price}")
            fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label="IV")
            surf2 = plot_iv_surface(ax2, deribit_puts, f"Deribit - {currency} - Puts")
            fig1.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label="IV")
            surf3 = plot_iv_surface(ax3, deribit_otm, f"Deribit - {currency} - OTM Puts and Calls\nTime (UTC): {utc_now} - Spot: {spot_price}")
            fig2.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, label="IV")

        fig1.subplots_adjust(left=0.02, right=0.98, bottom=0, top=1, wspace=0.1)

    # scatter1 = plot_iv_scatter(ax3, deribit_otm, f"Deribit - {currency} - OTM Puts and Calls")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()



