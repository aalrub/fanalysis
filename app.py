import io
import os
import json
import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
import finnhub

CACHE_DIR = "data_cache"
JSON_CATALOG_PATH = "tickers.json"
NEWS_CACHE_PATH = os.path.join(CACHE_DIR, "market_news.json")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

# --- Helper to load external CSS & HTML templates ---
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_html_template(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)

# Page Configuration
st.set_page_config(page_title="Multi-Instrument Analytics Engine", layout="wide")

# Load external CSS stylesheet
load_css("templates/style.css")

# --- Load & Save Ticker Catalog Helpers ---
def load_ticker_catalog(file_path):
    default_catalog = {
        "yahoo_tickers": {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
            "AMZN": "Amazon.com Inc.",
            "SPY": "S&P 500 ETF"
        },
        "fred_series": {
            "GDP": "Gross Domestic Product",
            "UNRATE": "Unemployment Rate",
            "CPIAUCSL": "Consumer Price Index",
            "FEDFUNDS": "Federal Funds Rate"
        }
    }
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading {file_path}, falling back to defaults: {e}")
            return default_catalog
    else:
        save_ticker_catalog(default_catalog, file_path)
        return default_catalog

def save_ticker_catalog(catalog, file_path):
    try:
        with open(file_path, "w") as f:
            json.dump(catalog, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to write to {file_path}: {e}")
        return False

# --- Helper for Simple Sentiment Evaluation ---
def estimate_sentiment(title):
    if not title:
        return "⚪ Neutral"
    title_lower = title.lower()
    positive_words = ['surge', 'jump', 'gain', 'growth', 'profit', 'rise', 'rally', 'beat', 'bull', 'upgrade', 'high', 'soar', 'record']
    negative_words = ['fall', 'drop', 'slump', 'decline', 'loss', 'plunge', 'warn', 'bear', 'downgrade', 'cut', 'risk', 'crisis', 'rout']
    
    pos_count = sum(word in title_lower for word in positive_words)
    neg_count = sum(word in title_lower for word in negative_words)
    
    if pos_count > neg_count:
        return "🟢 Positive"
    elif neg_count > pos_count:
        return "🔴 Negative"
    return "⚪ Neutral"

# --- Dynamic Plotly Chart Generator ---
def render_custom_chart(data, chart_type, title, labels=None):
    labels = labels or {"value": "Value / Price", "index": "Date"}
    
    if chart_type == "Line Chart":
        fig = px.line(data, title=title, labels=labels)
    elif chart_type == "Bar Chart":
        fig = px.bar(data, title=title, labels=labels, barmode="group")
    elif chart_type == "Area Chart":
        fig = px.area(data, title=title, labels=labels)
    elif chart_type == "Scatter Plot":
        fig = px.scatter(data, title=title, labels=labels)
    elif chart_type == "Pie Chart":
        if isinstance(data, pd.DataFrame):
            latest_row = data.iloc[-1].dropna()
            fig = px.pie(
                values=latest_row.values, 
                names=latest_row.index, 
                title=f"{title} (Latest Snapshot: {data.index[-1].strftime('%Y-%m-%d')})"
            )
        elif isinstance(data, pd.Series):
            fig = px.pie(
                values=data.values, 
                names=data.index, 
                title=title
            )
    else:
        fig = px.line(data, title=title, labels=labels)
    
    if chart_type != "Pie Chart":
        fig.update_layout(hovermode="x unified")
        
    return fig

# --- News Caching Helpers ---
def load_cached_news():
    if os.path.exists(NEWS_CACHE_PATH):
        try:
            with open(NEWS_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Failed to read news cache: {e}")
            return []
    return []

def save_cached_news(news_items):
    try:
        with open(NEWS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(news_items, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to update news JSON cache: {e}")

# --- App Navigation ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["📊 Analytics Dashboard", "⚙️ Manage Ticker Catalog"])

ticker_catalog = load_ticker_catalog(JSON_CATALOG_PATH)


# ==============================================================================
# PAGE 1: MANAGE TICKER CATALOG
# ==============================================================================
if page == "⚙️ Manage Ticker Catalog":
    st.title("⚙️ Manage Ticker Catalog (`tickers.json`)")
    st.markdown("Add, validate, and manage Yahoo Finance tickers and FRED Series IDs stored in your local `tickers.json` file.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("➕ Add Yahoo Finance Ticker")
        with st.form("add_yahoo_form", clear_on_submit=True):
            new_yf_symbol = st.text_input("Ticker Symbol (e.g., NVDA, TSLA, BTC-USD)").strip().upper()
            new_yf_name = st.text_input("Display Name (e.g., NVIDIA Corp)").strip()
            submit_yf = st.form_submit_button("Validate & Save Ticker")

            if submit_yf:
                if not new_yf_symbol or not new_yf_name:
                    st.error("Please fill in both the Symbol and Display Name fields.")
                else:
                    with st.spinner(f"Validating '{new_yf_symbol}' on Yahoo Finance..."):
                        try:
                            test_df = yf.download(new_yf_symbol, period="5d", progress=False)
                            if test_df.empty:
                                st.error(f"Could not validate '{new_yf_symbol}'. Please verify the symbol.")
                            else:
                                catalog = load_ticker_catalog(JSON_CATALOG_PATH)
                                catalog["yahoo_tickers"][new_yf_symbol] = new_yf_name
                                if save_ticker_catalog(catalog, JSON_CATALOG_PATH):
                                    st.success(f"Successfully saved **{new_yf_symbol}** ({new_yf_name}) to `tickers.json`!")
                                    st.rerun()
                        except Exception as err:
                            st.error(f"Validation failed for '{new_yf_symbol}': {err}")

    with col2:
        st.subheader("➕ Add FRED Series ID")
        with st.form("add_fred_form", clear_on_submit=True):
            new_fred_id = st.text_input("FRED Series ID (e.g., DGS10, M2SL)").strip().upper()
            new_fred_name = st.text_input("Display Name (e.g., 10-Year Treasury Rate)").strip()
            submit_fred = st.form_submit_button("Validate & Save FRED Series")

            if submit_fred:
                if not new_fred_id or not new_fred_name:
                    st.error("Please fill in both the Series ID and Display Name fields.")
                else:
                    with st.spinner(f"Validating '{new_fred_id}' on FRED..."):
                        try:
                            test_data = web.DataReader(new_fred_id, 'fred', start="2024-01-01", end="2024-01-10")
                            if test_data.empty:
                                st.error(f"Could not validate '{new_fred_id}'. Please verify the Series ID.")
                            else:
                                catalog = load_ticker_catalog(JSON_CATALOG_PATH)
                                catalog["fred_series"][new_fred_id] = new_fred_name
                                if save_ticker_catalog(catalog, JSON_CATALOG_PATH):
                                    st.success(f"Successfully saved **{new_fred_id}** ({new_fred_name}) to `tickers.json`!")
                                    st.rerun()
                        except Exception as err:
                            st.error(f"Validation failed for '{new_fred_id}': {err}")

    st.markdown("---")
    st.subheader("📋 Current Saved Catalog")
    cat_col1, cat_col2 = st.columns(2)
    with cat_col1:
        st.markdown("##### Yahoo Finance Tickers")
        st.json(ticker_catalog.get("yahoo_tickers", {}))
    with cat_col2:
        st.markdown("##### FRED Series IDs")
        st.json(ticker_catalog.get("fred_series", {}))


# ==============================================================================
# PAGE 2: ANALYTICS DASHBOARD
# ==============================================================================
elif page == "📊 Analytics Dashboard":
    render_html_template("templates/header.html")

    yahoo_options = [f"{symbol} - {name}" for symbol, name in ticker_catalog.get("yahoo_tickers", {}).items()]
    fred_options = [f"{symbol} - {name}" for symbol, name in ticker_catalog.get("fred_series", {}).items()]

    # --- Sidebar Inputs ---
    st.sidebar.header("⚙️ Data Selection Menu")

    finnhub_api_key = st.sidebar.text_input(
        "Finnhub API Key (for Historical News)", 
        type="password", 
        help="Get a free key from https://finnhub.io to query multi-year historical company news."
    ).strip()

    selected_yahoo = st.sidebar.multiselect(
        "Select Yahoo Finance Tickers",
        options=yahoo_options,
        default=yahoo_options[:2] if len(yahoo_options) >= 2 else yahoo_options
    )
    yahoo_tickers = [item.split(" - ")[0] for item in selected_yahoo]

    selected_fred = st.sidebar.multiselect(
        "Select FRED Data Series",
        options=fred_options,
        default=fred_options[:2] if len(fred_options) >= 2 else fred_options
    )
    fred_series_list = [item.split(" - ")[0] for item in selected_fred]

    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

    # --- Smart Caching Functions ---
    def load_series_from_json(file_path):
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                s = pd.Series(data, name="value")
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
            except Exception as e:
                st.warning(f"Could not read local cache file {file_path}: {e}")
        return None

    def save_series_to_json(series, file_path):
        try:
            clean_s = series.dropna().sort_index()
            data_dict = {date.strftime("%Y-%m-%d"): float(val) for date, val in clean_s.items()}
            with open(file_path, "w") as f:
                json.dump(data_dict, f, indent=2)
        except Exception as e:
            st.error(f"Failed to save cache to {file_path}: {e}")

    def get_yahoo_series_smart(ticker, start, end):
        file_path = os.path.join(CACHE_DIR, f"Yahoo_{ticker}.json")
        cached_s = load_series_from_json(file_path)
        req_start, req_end = pd.to_datetime(start), pd.to_datetime(end)

        if cached_s is not None and not cached_s.empty:
            if req_start >= cached_s.index.min() and req_end <= (cached_s.index.max() + pd.Timedelta(days=3)):
                return cached_s[(cached_s.index >= req_start) & (cached_s.index <= req_end)]

        raw_yf = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if not raw_yf.empty:
            new_s = raw_yf['Close'].squeeze() if 'Close' in raw_yf.columns else raw_yf.iloc[:, 0]
            new_s.index = pd.to_datetime(new_s.index)
            merged_s = new_s.combine_first(cached_s) if cached_s is not None else new_s
            save_series_to_json(merged_s, file_path)
            return merged_s[(merged_s.index >= req_start) & (merged_s.index <= req_end)]
            
        return cached_s

    def get_fred_series_smart(series_id, start, end):
        file_path = os.path.join(CACHE_DIR, f"FRED_{series_id}.json")
        cached_s = load_series_from_json(file_path)
        req_start, req_end = pd.to_datetime(start), pd.to_datetime(end)

        if cached_s is not None and not cached_s.empty:
            if req_start >= cached_s.index.min() and req_end <= (cached_s.index.max() + pd.Timedelta(days=7)):
                return cached_s[(cached_s.index >= req_start) & (cached_s.index <= req_end)]

        s_data = web.DataReader(series_id, 'fred', start, end)
        if not s_data.empty:
            new_s = s_data.iloc[:, 0]
            new_s.index = pd.to_datetime(new_s.index)
            merged_s = new_s.combine_first(cached_s) if cached_s is not None else new_s
            save_series_to_json(merged_s, file_path)
            return merged_s[(merged_s.index >= req_start) & (merged_s.index <= req_end)]
            
        return cached_s

    def fetch_multi_data(tickers, fred_series, start, end):
        combined_dfs = []
        for t in tickers:
            s = get_yahoo_series_smart(t, start, end)
            if s is not None and not s.empty:
                combined_dfs.append(pd.DataFrame({f"Yahoo_{t}": s}))

        for s_id in fred_series:
            s = get_fred_series_smart(s_id, start, end)
            if s is not None and not s.empty:
                combined_dfs.append(pd.DataFrame({f"FRED_{s_id}": s}))

        if not combined_dfs:
            raise ValueError("No data could be retrieved or loaded.")

        return pd.concat(combined_dfs, axis=1).ffill().dropna()

    def fetch_and_cache_market_news(tickers, start_date, end_date, api_key=""):
        cached_articles = load_cached_news()
        existing_urls = {item['URL'] for item in cached_articles if 'URL' in item}

        str_start = start_date.strftime("%Y-%m-%d")
        str_end = end_date.strftime("%Y-%m-%d")
        
        req_start = pd.to_datetime(start_date).tz_localize('UTC')
        req_end = pd.to_datetime(end_date).tz_localize('UTC') + pd.Timedelta(days=1)

        new_articles_added = 0

        # Finnhub API for Historical Date Range
        if api_key:
            try:
                finnhub_client = finnhub.Client(api_key=api_key)
                for ticker in tickers:
                    res = finnhub_client.company_news(ticker, _from=str_start, to=str_end)
                    for item in res:
                        link = item.get('url', '#')
                        if link not in existing_urls and link != '#':
                            title = item.get('headline', 'N/A')
                            publisher = item.get('source', 'Finnhub Source')
                            pub_dt = pd.to_datetime(item.get('datetime', 0), unit='s', utc=True)

                            new_entry = {
                                "Date": pub_dt.strftime('%Y-%m-%d %H:%M'),
                                "Ticker": ticker,
                                "Headline": title,
                                "Source": publisher,
                                "Sentiment": estimate_sentiment(title),
                                "URL": link
                            }
                            cached_articles.append(new_entry)
                            existing_urls.add(link)
                            new_articles_added += 1
            except Exception as e:
                st.error(f"Finnhub API Error: {e}")

        # yfinance Fallback
        else:
            for ticker in tickers:
                try:
                    t = yf.Ticker(ticker)
                    raw_news = t.news
                    if raw_news:
                        for item in raw_news:
                            link = item.get('link') or item.get('content', {}).get('canonicalUrl', {}).get('url', '#')
                            if link not in existing_urls and link != '#':
                                title = item.get('title') or item.get('content', {}).get('title', 'N/A')
                                publisher = item.get('publisher') or item.get('content', {}).get('provider', {}).get('displayName', 'Market Source')
                                pub_time = item.get('providerPublishTime') or item.get('content', {}).get('pubDate')
                                
                                pub_dt = pd.to_datetime(pub_time, unit='s', utc=True) if isinstance(pub_time, (int, float)) else pd.to_datetime(pub_time or 'today', utc=True)

                                new_entry = {
                                    "Date": pub_dt.strftime('%Y-%m-%d %H:%M'),
                                    "Ticker": ticker,
                                    "Headline": title,
                                    "Source": publisher,
                                    "Sentiment": estimate_sentiment(title),
                                    "URL": link
                                }
                                cached_articles.append(new_entry)
                                existing_urls.add(link)
                                new_articles_added += 1
                except Exception as e:
                    st.warning(f"Could not fetch news via yfinance for {ticker}: {e}")

        if new_articles_added > 0:
            save_cached_news(cached_articles)

        filtered_news = []
        for item in cached_articles:
            if item.get("Ticker") in tickers:
                item_dt = pd.to_datetime(item.get("Date"), utc=True)
                if req_start <= item_dt <= req_end:
                    filtered_news.append(item)

        if filtered_news:
            news_df = pd.DataFrame(filtered_news)
            news_df['dt_sort'] = pd.to_datetime(news_df['Date'])
            news_df = news_df.sort_values(by="dt_sort", ascending=False).drop(columns=["dt_sort"])
            return news_df

        return pd.DataFrame()

    # --- Fetch Execution ---
    if st.sidebar.button("🚀 Fetch and Process Data"):
        if not yahoo_tickers and not fred_series_list:
            st.error("Please select at least one Yahoo Finance Ticker or FRED Series ID.")
        else:
            with st.spinner("Processing local JSON cache & updating web data..."):
                try:
                    df = fetch_multi_data(yahoo_tickers, fred_series_list, start_date, end_date)
                    st.session_state['data'] = df
                    st.session_state['active_tickers'] = yahoo_tickers
                    st.session_state['start_date'] = start_date
                    st.session_state['end_date'] = end_date
                    st.session_state['finnhub_key'] = finnhub_api_key
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

    # --- Dashboard Content ---
    if 'data' in st.session_state:
        df = st.session_state['data']

        st.subheader("📋 Dataset Overview")
        st.dataframe(df.tail(10), use_container_width=True)

        st.subheader("📈 Dynamic Visualization")
        
        c_col1, c_col2 = st.columns([3, 1])
        with c_col1:
            selected_series = st.multiselect(
                "Select Series for Main Chart", 
                options=df.columns.tolist(), 
                default=df.columns.tolist(), 
                key="main_chart_select"
            )
        with c_col2:
            main_chart_style = st.selectbox(
                "Main Chart Style", 
                ["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot", "Pie Chart"],
                key="main_chart_style"
            )

        if selected_series:
            fig_main = render_custom_chart(
                df[selected_series], 
                main_chart_style, 
                title=f"Raw Price/Value Comparison ({main_chart_style})", 
                labels={"value": "Level / Price", "index": "Date"}
            )
            st.plotly_chart(fig_main, use_container_width=True)

        # --- Descriptive Statistics & Data Analysis Section ---
        st.markdown("---")
        st.subheader("📊 Descriptive Statistics & Exploratory Data Analysis")

        stats_series = st.multiselect(
            "Select Variables for Statistical Summary", 
            options=df.columns.tolist(), 
            default=df.columns.tolist(),
            key="stats_series_select"
        )

        if stats_series:
            sub_stats_df = df[stats_series]
            
            # Compute Descriptive Metrics
            metrics_list = []
            for col in sub_stats_df.columns:
                s = sub_stats_df[col].dropna()
                
                # Mode calculation
                mode_res = s.mode()
                mode_val = round(float(mode_res.iloc[0]), 4) if not mode_res.empty else np.nan

                # Quartiles & Range
                q1 = s.quantile(0.25)
                q2 = s.quantile(0.50)  # Median
                q3 = s.quantile(0.75)
                iqr = q3 - q1

                metrics_list.append({
                    "Variable": col,
                    "Count": len(s),
                    "Mean": s.mean(),
                    "Median (Q2)": q2,
                    "Mode": mode_val,
                    "Variance": s.var(),
                    "Std Dev": s.std(),
                    "Min": s.min(),
                    "25% (Q1)": q1,
                    "75% (Q3)": q3,
                    "Max": s.max(),
                    "Range": s.max() - s.min(),
                    "IQR": iqr
                })

            summary_df = pd.DataFrame(metrics_list).set_index("Variable")

            # Display Data Table
            st.markdown("##### Comprehensive Descriptive Metrics Table")
            st.dataframe(
                summary_df.style.format({
                    "Count": "{:,.0f}",
                    "Mean": "{:,.4f}",
                    "Median (Q2)": "{:,.4f}",
                    "Mode": "{:,.4f}",
                    "Variance": "{:,.4f}",
                    "Std Dev": "{:,.4f}",
                    "Min": "{:,.4f}",
                    "25% (Q1)": "{:,.4f}",
                    "75% (Q3)": "{:,.4f}",
                    "Max": "{:,.4f}",
                    "Range": "{:,.4f}",
                    "IQR": "{:,.4f}"
                }),
                use_container_width=True
            )

            # Interactive Distribution Visualizations
            st.markdown("##### Visualizing Distributions & Dispersion")
            viz_tab1, viz_tab2 = st.tabs(["📦 Quartile & Outlier Box Plots", "📊 Histogram & Density Plot"])

            with viz_tab1:
                fig_box = px.box(
                    sub_stats_df, 
                    title="Box Plot Analysis (Quartiles, Median & Outliers)",
                    points="outliers"
                )
                fig_box.update_layout(yaxis_title="Value / Level")
                st.plotly_chart(fig_box, use_container_width=True)

            with viz_tab2:
                selected_hist_col = st.selectbox("Select Variable to Plot Histogram", options=stats_series)
                fig_hist = px.histogram(
                    sub_stats_df, 
                    x=selected_hist_col, 
                    marginal="rug", 
                    nbins=40,
                    title=f"Histogram & Distribution Density for {selected_hist_col}",
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        # Secondary Additional Chart Section
        st.markdown("---")
        with st.expander("➕ Add Additional Analytical Chart", expanded=True):
            sec_col1, sec_col2 = st.columns([2, 1])
            with sec_col1:
                chart_type = st.selectbox(
                    "Select Secondary Analysis Metric",
                    [
                        "Normalized Performance (Base = 100)",
                        "Percentage Daily Returns",
                        "Rolling 30-Day Volatility (Std Dev)",
                        "Correlation Heatmap"
                    ]
                )
            with sec_col2:
                sec_chart_style = st.selectbox(
                    "Secondary Chart Display Style",
                    ["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot", "Pie Chart"],
                    key="sec_chart_style",
                    disabled=(chart_type == "Correlation Heatmap")
                )

            secondary_series = st.multiselect(
                "Select Series for Secondary Chart", 
                options=df.columns.tolist(), 
                default=df.columns.tolist(),
                key="secondary_chart_select"
            )

            if secondary_series:
                sub_df = df[secondary_series]

                if chart_type == "Normalized Performance (Base = 100)":
                    norm_df = (sub_df / sub_df.iloc[0]) * 100
                    fig_sec = render_custom_chart(
                        norm_df, 
                        sec_chart_style, 
                        title="Normalized Performance Growth (Rebased to 100)", 
                        labels={"value": "Indexed Performance (100 = Start)"}
                    )
                    st.plotly_chart(fig_sec, use_container_width=True)

                elif chart_type == "Percentage Daily Returns":
                    returns_df = sub_df.pct_change().dropna() * 100
                    fig_sec = render_custom_chart(
                        returns_df, 
                        sec_chart_style, 
                        title="Daily Percentage Returns (%)", 
                        labels={"value": "Daily Return (%)"}
                    )
                    st.plotly_chart(fig_sec, use_container_width=True)

                elif chart_type == "Rolling 30-Day Volatility (Std Dev)":
                    vol_df = sub_df.pct_change().rolling(window=30).std() * np.sqrt(252) * 100
                    fig_sec = render_custom_chart(
                        vol_df, 
                        sec_chart_style, 
                        title="30-Day Rolling Annualized Volatility (%)", 
                        labels={"value": "Annualized Volatility (%)"}
                    )
                    st.plotly_chart(fig_sec, use_container_width=True)

                elif chart_type == "Correlation Heatmap":
                    corr_matrix = sub_df.corr()
                    fig_sec = px.imshow(
                        corr_matrix, 
                        text_auto=".2f", 
                        aspect="auto", 
                        color_continuous_scale="RdBu_r",
                        title="Cross-Instrument Correlation Matrix"
                    )
                    st.plotly_chart(fig_sec, use_container_width=True)

        # --- High Impact Market News Section ---
        st.markdown("---")
        st.subheader("📰 High-Impact Market & Asset News")
        
        active_t = st.session_state.get('active_tickers', yahoo_tickers)
        if active_t:
            with st.spinner("Checking & updating local market news cache..."):
                news_df = fetch_and_cache_market_news(
                    active_t, 
                    st.session_state.get('start_date', start_date), 
                    st.session_state.get('end_date', end_date),
                    api_key=st.session_state.get('finnhub_key', finnhub_api_key)
                )

            if not news_df.empty:
                st.markdown(f"Displaying stored & fetched headlines for selected assets (**{', '.join(active_t)}**):")
                st.dataframe(
                    news_df,
                    column_config={
                        "URL": st.column_config.LinkColumn("Read Article", display_text="Open Link"),
                        "Date": st.column_config.TextColumn("Published Time"),
                        "Sentiment": st.column_config.TextColumn("Headline Sentiment")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No matching headlines were found within the specified date range for the active tickers.")
        else:
            st.info("Select active Yahoo Finance tickers to display high-impact news coverage.")

        # --- OLS Regression Analysis ---
        st.markdown("---")
        st.subheader("📐 Multiple Linear Regression (OLS) & Analysis")
        if len(df.columns) >= 2:
            reg1, reg2 = st.columns(2)
            dep = reg1.selectbox("Dependent Variable (Y)", options=df.columns.tolist())
            indep = reg2.multiselect("Independent Variables (X)", options=[c for c in df.columns if c != dep])
            
            if dep and indep:
                Y = df[dep]
                X = sm.add_constant(df[indep])
                model = sm.OLS(Y, X).fit()
                
                dw_val = sm.stats.stattools.durbin_watson(model.resid)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R-Squared ($R^2$)", f"{model.rsquared:.4f}")
                m2.metric("Adj. R-Squared", f"{model.rsquared_adj:.4f}")
                m3.metric("F-Statistic", f"{model.fvalue:.2f}")
                m4.metric("Durbin-Watson", f"{dw_val:.3f}")

                if dw_val < 1.0:
                    st.markdown("""
                        <div class="alert-box alert-warning">
                            ⚠️ <b>Autocorrelation Alert:</b> Durbin-Watson score is extremely low (&lt; 1.0). Residuals exhibit strong positive autocorrelation across time periods. Consider fitting the model using first differences (<code>df.diff()</code>) or returns.
                        </div>
                    """, unsafe_allow_html=True)
                if model.condition_number > 1e4:
                    st.markdown(f"""
                        <div class="alert-box alert-info">
                            💡 <b>Multicollinearity Warning:</b> High condition number ({model.condition_number:.2e}). Macroeconomic variables may be highly collinear.
                        </div>
                    """, unsafe_allow_html=True)

                r_tab1, r_tab2, r_tab3, r_tab4, r_tab5 = st.tabs([
                    "📊 Parameter Estimates Table",
                    "🎯 Actual vs Fitted Scatter",
                    "📈 Feature Relationships",
                    "⏱️ Time Series Fit",
                    "📄 Raw Statsmodels Output"
                ])

                with r_tab1:
                    conf_int = model.conf_int()
                    conf_int.columns = ["Conf. Interval Lower", "Conf. Interval Upper"]
                    
                    coef_df = pd.DataFrame({
                        "Variable": model.params.index,
                        "Coefficient": model.params.values,
                        "Std. Error": model.bse.values,
                        "t-Statistic": model.tvalues.values,
                        "p-Value": model.pvalues.values,
                        "95% CI Lower": conf_int["Conf. Interval Lower"].values,
                        "95% CI Upper": conf_int["Conf. Interval Upper"].values
                    })
                    
                    def highlight_significant(val):
                        if isinstance(val, float) and val < 0.05:
                            return 'background-color: rgba(76, 175, 80, 0.2); font-weight: bold;'
                        return ''

                    st.markdown("##### Model Coefficient Estimates & Hypothesis Testing")
                    st.dataframe(
                        coef_df.style.map(highlight_significant, subset=["p-Value"])
                        .format({
                            "Coefficient": "{:.4f}",
                            "Std. Error": "{:.4f}",
                            "t-Statistic": "{:.3f}",
                            "p-Value": "{:.4e}",
                            "95% CI Lower": "{:.4f}",
                            "95% CI Upper": "{:.4f}"
                        }),
                        hide_index=True,
                        use_container_width=True
                    )

                with r_tab2:
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=model.fittedvalues,
                        y=Y,
                        mode='markers',
                        name='Observations',
                        marker=dict(color='#1f77b4', opacity=0.7)
                    ))
                    min_val = min(model.fittedvalues.min(), Y.min())
                    max_val = max(model.fittedvalues.max(), Y.max())
                    fig_scatter.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Ideal Fit Line (45°)',
                        line=dict(color='red', dash='dash')
                    ))
                    fig_scatter.update_layout(
                        title=f"Actual Y vs Fitted Y ({dep})",
                        xaxis_title="Predicted Y (Fitted Values)",
                        yaxis_title="Actual Y (Observed Values)"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                with r_tab3:
                    selected_x_scatter = st.selectbox("Select X feature to plot against Y", options=indep)
                    fig_feat = px.scatter(
                        df,
                        x=selected_x_scatter,
                        y=dep,
                        trendline="ols",
                        trendline_color_override="red",
                        title=f"Scatter Plot: {dep} (Y) vs {selected_x_scatter} (X)"
                    )
                    st.plotly_chart(fig_feat, use_container_width=True)

                with r_tab4:
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(x=df.index, y=Y, mode='lines', name='Actual Y'))
                    fig_ts.add_trace(go.Scatter(x=df.index, y=model.fittedvalues, mode='lines', name='Fitted Y', line=dict(dash='dash')))
                    fig_ts.update_layout(title=f"Time Series: Actual vs Fitted ({dep})", hovermode="x unified")
                    st.plotly_chart(fig_ts, use_container_width=True)

                with r_tab5:
                    st.text(model.summary())
        else:
            st.info("OLS Regression requires at least 2 variables in the active dataset.")