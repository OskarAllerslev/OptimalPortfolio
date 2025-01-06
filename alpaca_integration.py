# alpaca_integration.py
import streamlit as st
import alpaca_trade_api as tradeapi

def connect_alpaca_and_get_account(api_key: str, secret_key: str, base_url: str):
    """
    Connect to Alpaca and return the API object and account.
    Raises an exception if credentials are invalid or if something goes wrong.
    """
    api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
    account = api.get_account()
    return api, account

def invest_in_portfolio(api, account, assets, weights, portion):
    """
    portion is a float between 0.0 and 1.0 indicating the fraction of the
    account's equity to invest.
    """
    # Convert equity string to float
    equity = float(account.equity)
    amount_to_invest = equity * portion

    st.write(f"Account equity: {equity}")
    st.write(f"Amount to invest: {amount_to_invest}")

    # For each asset, buy `weight * amount_to_invest` worth (a notional order)
    for asset, w in zip(assets, weights):
        notional = round(amount_to_invest * w, 2)  # Limit to 2 decimal places
        # if notional is very small, skip to avoid errors
        if notional < 1.0:
            continue

        # NOTE: Alpaca only supports US tickers. If your asset list has ".L" (LSE) 
        # or other non-US tickers, you'll need to map them to US symbols or skip them. 
        # For demonstration, let's just strip ".L" but be aware that might not work
        # if your symbols aren't actually traded in the US.
        symbol = asset.replace(".L", "")

        st.write(f"Placing market buy for {symbol} with notional = {notional:.2f}")

        try:
            order = api.submit_order(
                symbol=symbol,
                notional=notional,   # dollar-based order
                side='buy',
                type='market',
                time_in_force='day'
            )
            st.write(f"Order for {symbol} submitted: {order}")
        except Exception as e:
            st.error(f"Failed to place order for {symbol}: {e}")