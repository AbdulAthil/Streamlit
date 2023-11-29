import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly import graph_objs as go
from datetime import date

st.title("Stock Price Analysis App")
st.write("""
* By using this app,we can view and analyse the stock price data
of various global companies such as Google, Apple, Amazon and Microsoft etc..
from January 2023 to till date.
* By using this app, we can analyse the stock price details of any company.
* And we can compare the stocks between two companies
""")
# st.divider()

tab11, tab12 = st.tabs(["View Stocks", "Compare Stocks"])
with tab11:
    # View Stocks
    own = st.checkbox("Click here to Add on Your own")
    if own:
        text = st.text_input("Enter which company stocks you want to analyse")
        st.write("###### Please select", text, "from below")
    else:
        text = "None"

    stocks = ("Select", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA",
              "SMSN.IL", "SONY", "XIACY", "SPOT", "NFLX", "INTC", "NVDA", text)
    selected_stocks = st.selectbox("Select dataset for analysis", stocks)
    if selected_stocks == "None":
        st.write("Please select a dataset from above")
    elif selected_stocks == "Select":
        st.write("Select any dataset from above")
    else:
        # st.write("Selected dataset: ", selected_stocks)
        st.divider()
        st.subheader(f"{selected_stocks} stock values")
        # define a ticker symbol
        ticker_symbol = selected_stocks
        # get data on this ticker
        ticker_data = yf.Ticker(ticker_symbol)
        # get the historical prices for this ticker
        START = "2023-1-1"
        END = date.today().strftime("%Y-%m-%d")
        tickerDf = ticker_data.history(period="id", start=START, end=END)
        st.dataframe(tickerDf)
        # Visualization
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Open", "High ", "Low", "Close", "Volume"])
        with tab1:
            st.write("#### Opening Price values")
            st.line_chart(tickerDf.Open, use_container_width=True,)
        with tab2:
            st.write("#### Highest Price values")
            st.line_chart(tickerDf.High, use_container_width=True,)
        with tab3:
            st.write("#### Lowest Price values")
            st.line_chart(tickerDf.Low, use_container_width=True,)
        with tab4:
            st.write("#### Closing Price values")
            st.line_chart(tickerDf.Close, use_container_width=True,)
        with tab5:
            st.write("#### Volume")
            st.line_chart(tickerDf.Volume, use_container_width=True,)
        st.divider()

with tab12:
    # Comparison
    # Default
    st.subheader("Compare two company stocks")
    own = st.checkbox("Click here to Add on your own")
    col1, col2 = st.columns(2)
    # First company
    with col1:
        if own:
            text1 = st.text_input("Enter First Company")
            st.write("###### Please select", text1, "from below")
        else:
            text1 = "None"
        stocks1 = ("Select", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA",
                  "SMSN.IL", "SONY", "XIACY", "SPOT", "NFLX", "INTC", "NVDA", text1)
        compare_stocks1 = st.selectbox("Select first dataset", stocks1)
        # st.write("Selected dataset: ", compare_stocks1)
        if compare_stocks1 == "Select":
            st.write("Select a dataset from above")
        st.divider()
        st.write(f"##### {compare_stocks1} stock values")
        # define a ticker symbol
        ticker_symbol1 = compare_stocks1
        # get data on this ticker
        ticker_data1 = yf.Ticker(ticker_symbol1)
        # get the historical prices for this ticker
        START = "2023-1-1"
        END = date.today().strftime("%Y-%m-%d")
        tickerDf1 = ticker_data1.history(period="id", start=START, end=END)
        st.dataframe(tickerDf1.tail())
        st.divider()

    # Second company
    with col2:
        if own:
            text2 = st.text_input("Enter Second Company")
            st.write("###### Please select", text2, "from below")
        else:
            text2 = "None"
        stocks2 = ("Select", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA",
                  "SMSN.IL", "SONY", "XIACY", "SPOT", "NFLX", "INTC", "NVDA", text2)
        compare_stocks2 = st.selectbox("Select second dataset", stocks2)
        # st.write("Selected dataset: ", compare_stocks2)
        if compare_stocks2 == "Select":
            st.write("Select a dataset from above")
        st.divider()
        st.write(f"##### {compare_stocks2} stock values")
        # define a ticker symbol
        ticker_symbol2 = compare_stocks2
        # get data on this ticker
        ticker_data2 = yf.Ticker(ticker_symbol2)
        # get the historical prices for this ticker
        START = "2023-1-1"
        END = date.today().strftime("%Y-%m-%d")
        tickerDf2 = ticker_data2.history(period="id", start=START, end=END)
        st.dataframe(tickerDf2.tail())
        st.divider()

    # Visualization
    compare_companies = [compare_stocks1, compare_stocks2]
    cols = tickerDf1.columns
    # st.write(cols)
    select_col = st.selectbox("Select which column you want to compare", cols)

    tab1, tab2, tab3, tab4 = st.tabs(["Sum", "Mean", "Max", "Min"])
    with tab1:
        # Sum value
        c1_sum = tickerDf1[select_col].sum()
        st.write(compare_stocks1, "-", c1_sum)
        c2_sum = tickerDf2[select_col].sum()
        st.write(compare_stocks2, "-", c2_sum)
        compare_sum = [c1_sum, c2_sum]
        st.caption("Comparison")
        df = pd.DataFrame(compare_sum, index=compare_companies)
        st.bar_chart(df, use_container_width=True)

    with tab2:
        # Mean value
        c1_mean = tickerDf1[select_col].mean()
        st.write(compare_stocks1, "-", c1_mean)
        c2_mean = tickerDf2[select_col].mean()
        st.write(compare_stocks2, "-", c2_mean)
        compare_mean = [c1_mean, c2_mean]
        st.caption("Comparison")
        df = pd.DataFrame(compare_mean, index=compare_companies)
        st.bar_chart(df, use_container_width=True)

    with tab3:
        # Sum value
        c1_max = tickerDf1[select_col].max()
        st.write(compare_stocks1, "-", c1_max)
        c2_max = tickerDf2[select_col].max()
        st.write(compare_stocks2, "-", c2_max)
        compare_max = [c1_max, c2_max]
        st.caption("Comparison")
        df = pd.DataFrame(compare_max, index=compare_companies)
        st.bar_chart(df, use_container_width=True)

    with tab4:
        # Sum value
        c1_min = tickerDf1[select_col].min()
        st.write(compare_stocks1, "-", c1_min)
        c2_min = tickerDf2[select_col].min()
        st.write(compare_stocks2, "-", c2_min)
        compare_min = [c1_min, c2_min]
        st.caption("Comparison")
        df = pd.DataFrame(compare_min, index=compare_companies)
        st.bar_chart(df, use_container_width=True)

