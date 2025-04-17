
import streamlit as st
from stock_app import run_stock_app
from bitcoin_app import run_bitcoin_app  # Ensure this module is created similarly to stock_app

def main():
    st.set_page_config(page_title="FincastAI", page_icon="📊", layout="wide")

    st.title("📊 FincastAI")
    st.markdown("""
    Welcome to **FincastAI** – your comprehensive platform for stock and Bitcoin price prediction.
    Navigate through the options below to explore predictive analytics for financial markets.
    """)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_selection = st.sidebar.radio("Go to", ["Home", "Stock Predictor", "Bitcoin Predictor"])

    if app_selection == "Home":
        st.subheader("Overview")
        st.markdown("""
        **FincastAI** leverages advanced machine learning models to forecast future trends in stock and cryptocurrency markets.
        Select an option from the sidebar to begin your analysis.
        """)
    elif app_selection == "Stock Predictor":
        run_stock_app()
    elif app_selection == "Bitcoin Predictor":
        run_bitcoin_app()

if __name__ == "__main__":
    main()