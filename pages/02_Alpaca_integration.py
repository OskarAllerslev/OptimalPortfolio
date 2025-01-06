# pages/02_Alpaca_integration.py

import streamlit as st
from alpaca_integration import connect_alpaca_and_get_account, invest_in_portfolio

# Optional: set a custom page config
st.set_page_config(page_title="Alpaca Integration")

st.title("Alpaca Integration")

# 1) Check if we have an optimized portfolio in st.session_state
if "chosen_assets" not in st.session_state or "chosen_weights" not in st.session_state:
    st.warning("No optimized portfolio found. Please run the optimization on the main page first.")
    st.stop()

chosen_assets = st.session_state["chosen_assets"]
chosen_weights = st.session_state["chosen_weights"]

st.subheader("Chosen Portfolio From Optimization")
st.write("**Assets:**", chosen_assets)
st.write("**Weights:**", chosen_weights)

# 2) Ask for Alpaca credentials
st.subheader("Enter Alpaca Credentials (Paper Trading)")
api_key = st.text_input("API Key")
secret_key = st.text_input("Secret Key", type="password")

# We'll default to paper environment for now
base_url = "https://paper-api.alpaca.markets"

if st.button("Connect to Alpaca"):
    if api_key and secret_key:
        try:
            api, account = connect_alpaca_and_get_account(api_key, secret_key, base_url)
            st.session_state["alpaca_api"] = api
            st.session_state["alpaca_account"] = account
            st.success("Successfully connected to Alpaca (Paper)!")
        except Exception as e:
            st.error(f"Error connecting to Alpaca: {e}")
    else:
        st.warning("Please provide both API Key and Secret Key.")

# 3) If connected, show account info & let user choose how much to invest
if "alpaca_api" in st.session_state and "alpaca_account" in st.session_state:
    account = st.session_state["alpaca_account"]
    st.subheader("Account Info")
    st.write(f"**Status**: {account.status}")
    st.write(f"**Equity**: {account.equity}")
    st.write(f"**Cash**: {account.cash}")

    st.subheader("Invest in Optimized Portfolio")
    portion = st.slider("Fraction of account equity to invest", 0.0, 1.0, 0.3)

    if st.button("Invest Now"):
        invest_in_portfolio(
            api=st.session_state["alpaca_api"],
            account=account,
            assets=chosen_assets,
            weights=chosen_weights,
            portion=portion
        )
