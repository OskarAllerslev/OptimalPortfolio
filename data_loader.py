import yfinance as yf
import streamlit as st

@st.cache_data
def download_data(lookback_years: int):
    """Download UCITS tickers from yfinance."""
    # ucits_tickers = [
    #     "SWDA.L", "SSAC.L", "EMIM.L", "CSP1.L", "VUSA.L", "VUAA.L", "EQQQ.L", "VFEM.L",
    #     "VWRL.L", "XDWD.L", "IMEU.L", "SWRD.L", "IEMA.L", "VERX.L", "SPY5.L", "IUSA.L",
    #     "IWVL.L", "IWMO.L", "INRG.L", "HEAL.L", "GOAT.L", "GDX.L", "IUIT.L", "BNKS.L",
    #     "HMWO.L", "HMEF.L", "AGGG.L", "IBTS.L", "IBTM.L",
    #     "LQDE.L", "IEMB.L", "IGLH.L", "AGGU.L",
    #     "SGLN.L", "SSLN.L", "SGLD.L", "SSIL.L",
    #     "XGIG.L", "IUKP.L", "IWDP.L", "SPY4.L", "VHYL.L",
    #     "VMID.L", "XLFS.L", "CSPX.L", "V3AA.L", "V3AM.L",
    #     "SGLP.L"
    # ]

    #below us tickers
    ucits_tickers = [
        "SPY",  # S&P 500 ETF
        "QQQ",  # Nasdaq 100 ETF
        "IWM",  # Russell 2000 ETF
        "VEA",  # Vanguard FTSE Developed Markets
        "EFA",  # iShares MSCI EAFE
        "AGG",  # iShares Core US Aggregate Bond
        "TLT",  # iShares 20+ Year Treasury
        "LQD",  # iShares Investment-Grade Corporate Bonds
        "HYG",  # iShares High-Yield Corporate Bond
        "GLD",  # SPDR Gold
        "SLV",  # iShares Silver
        "GDX",  # Gold Miners
        "VNQ",  # Vanguard REIT
        "XLE",  # Energy sector
        "XLF",  # Financials sector
        "XLK",  # Technology sector
        "XLV",  # Healthcare sector
        "XLY",  # Consumer Discretionary
        "XLP",  # Consumer Staples
        "XLB",  # Materials
        # add more if you like, but 20 is often enough for a broad sample
    ]

    raw_data = yf.download(
        ucits_tickers, 
        period=f"{lookback_years}y", 
        interval="1d"
    )["Adj Close"]
    return raw_data

@st.cache_data
def download_spy_data(lookback_years: int):
    """Download SPY data as benchmark."""
    spy_data = yf.download(
        "SPY", 
        period=f"{lookback_years}y", 
        interval="1d"
    )["Adj Close"]
    return spy_data
