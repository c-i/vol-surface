import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
# import asyncio


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

    # IV: Deribit’s mark_iv looks like percent (e.g., 66.28). Keep as % or convert to decimal:
    df["iv"] = df["mark_iv"]            # as percent
    # df["iv"] = df["mark_iv"] / 100.0   # uncomment for decimal (0.6628)

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
    print(deribit_puts.head())

    fig = plt.figure(figsize=(9, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = plot_iv_surface(ax1, deribit_calls, "Deribit - Calls")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label="IV (%)")

    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = plot_iv_surface(ax2, deribit_puts, "Deribit - Puts")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label="IV (%)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()



# must calculate IV manually using vollib or scipy, deribit uses same mark iv estimate for puts and calls