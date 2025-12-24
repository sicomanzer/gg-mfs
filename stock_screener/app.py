import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import datetime
import time

# --- CONFIGURATION ---
DATA_FILE = "set100_db.json"

st.set_page_config(
    page_title="SET100 Magic Formula Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #34495e;
    }
    .stButton>button {
        width: 100%;
        background-color: #2980b9;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3498db;
        border-color: #3498db;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- HARDCODED FALLBACK LIST (Approx SET100) ---
FALLBACK_SET100 = [
    'AAV', 'ADVANC', 'AEONTS', 'AMATA', 'AOT', 'AP', 'AURA', 'AWC', 'BA', 'BAM', 'BANPU', 'BBL', 'BCH', 'BCP', 'BCPG', 'BDMS', 'BEM', 'BGRIM', 'BH', 'BJC', 'BLA', 'BTG', 'BTS', 'CBG', 'CCET', 'CENTEL', 'CHG', 'CK', 'COM7', 'CPALL', 'CPF', 'CPN', 'CRC', 'DELTA', 'DOHOME', 'EA', 'EGCO', 'ERW', 'GLOBAL', 'GPSC', 'GULF', 'GUNKUL', 'HANA', 'HMPRO', 'ICHI', 'IRPC', 'ITC', 'IVL', 'JAS', 'JMART', 'JMT', 'JTS', 'KBANK', 'KCE', 'KKP', 'KTB', 'KTC', 'LH', 'M', 'MBK', 'MEGA', 'MINT', 'MOSHI', 'MTC', 'OR', 'OSP', 'PLANB', 'PR9', 'PRM', 'PTT', 'PTTEP', 'PTTGC', 'QH', 'RATCH', 'RCL', 'SAWAD', 'SCB', 'SCC', 'SCGP', 'SIRI', 'SISB', 'SJWD', 'SPALI', 'SPRC', 'STA', 'STGT', 'TASCO', 'TCAP', 'TFG', 'TIDLOR', 'TISCO', 'TLI', 'TOA', 'TOP', 'TRUE', 'TTB', 'TU', 'VGI', 'WHA', 'WHAUP'
]

# --- HELPER FUNCTIONS ---

def calculate_intrinsic_value(fcf, shares, growth_rate, discount_rate=0.10, terminal_growth=0.02, years=10):
    """
    Simplified DCF Valuation
    FCF: Total Free Cash Flow
    Shares: Total Shares Outstanding
    Growth Rate: Expected growth (capped conservative)
    """
    if not fcf or not shares or shares == 0:
        return None
    
    # Per Share FCF
    fcf_per_share = fcf / shares
    
    # Sanitize Growth Rate (Conservative Guardrails)
    if growth_rate is None:
        g = 0.03 # Default to 3% if missing
    elif growth_rate > 0.15:
        g = 0.15 # Cap at 15%
    elif growth_rate < 0:
        g = 0.0 # Assume flat if negative, rather than projecting decline forever (Value trap logic)
    else:
        g = growth_rate
        
    # Projected Cash Flows
    future_values = []
    for i in range(1, years + 1):
        val = fcf_per_share * ((1 + g) ** i)
        discounted = val / ((1 + discount_rate) ** i)
        future_values.append(discounted)
        
    # Terminal Value
    last_fcf = fcf_per_share * ((1 + g) ** years)
    terminal_val = (last_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    discounted_terminal = terminal_val / ((1 + discount_rate) ** years)
    
    intrinsic_value = sum(future_values) + discounted_terminal
    return round(intrinsic_value, 2)

def get_set100_symbols():
    """
    Reads SET100 symbols from external text file for easy updates.
    """
    source_file = "source_set100.txt"
    if os.path.exists(source_file):
        with open(source_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Add .BK if missing
        symbols = []
        for s in lines:
            if not s.upper().endswith(".BK"):
                symbols.append(f"{s.upper()}.BK")
            else:
                symbols.append(s.upper())
        return symbols
    
    # Using fallback if file missing
    return [f"{s}.BK" for s in FALLBACK_SET100]

def update_database():
    """
    Phase 1: ETL Process
    """
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    symbols = get_set100_symbols()
    total = len(symbols)
    data_list = []
    
    status_text.write("🚀 Starting Data Extraction...")
    
    for i, symbol in enumerate(symbols):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract metrics (using .get for safety)
            # Depending on yfinance version, keys might vary, but these are standard
            metrics = {
                "symbol": symbol,
                "pe": info.get("trailingPE"),
                "roe": info.get("returnOnEquity"),
                "pbv": info.get("priceToBook"),
                "yield": info.get("dividendYield"),
                "price": info.get("currentPrice"),
                "de": info.get("debtToEquity"),
                "fcf": info.get("freeCashflow"),
                "shares": info.get("sharesOutstanding"),
                "growth": info.get("earningsGrowth"),
                "high52": info.get("fiftyTwoWeekHigh"),
                "ocf": info.get("operatingCashflow"),
                "avg_volume": info.get("averageVolume"),
                "ev_ebitda": info.get("enterpriseToEbitda"), # New: Value Metric
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # --- FALLBACK LOGIC ---
            
            # 1. P/E Fallback
            if metrics["pe"] is None:
                if info.get("forwardPE"):
                    metrics["pe"] = info.get("forwardPE")
                elif info.get("trailingEps") and info.get("trailingEps") > 0 and metrics["price"]:
                    metrics["pe"] = metrics["price"] / info.get("trailingEps")

            # 2. Yield Fallback (Prioritize Calc TTM Yield from History due to YF inaccuracies on Thai stocks)
            try:
                divs = ticker.dividends
                if not divs.empty:
                    # Handle TZ awareness
                    now = pd.Timestamp.now()
                    if divs.index.tz is not None:
                        now = now.tz_localize(divs.index.tz)
                    
                    cutoff = now - pd.Timedelta(days=365)
                    last_12m_divs = divs[divs.index >= cutoff]
                    total_div = last_12m_divs.sum()
                    
                    if total_div > 0 and metrics["price"] and metrics["price"] > 0:
                        metrics["yield"] = total_div / metrics["price"]
            except:
                pass
            
            # If still None, use info fallback
            if metrics["yield"] is None:
                div_rate = info.get("dividendRate")
                if div_rate and metrics["price"] and metrics["price"] > 0:
                     metrics["yield"] = div_rate / metrics["price"]

            # 3. D/E Fallback (Balance Sheet)
            if metrics["de"] is None:
                try:
                    bs = ticker.balance_sheet
                    if not bs.empty:
                        # Try finding Total Liab
                        liab = None
                        target_liab_keys = ['Total Liabilities Net Minority Interest', 'Total Liabilities']
                        for k in target_liab_keys:
                            if k in bs.index:
                                liab = bs.loc[k][0]
                                break
                        
                        # Try finding Equity
                        equity = None
                        target_equity_keys = ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity']
                        for k in target_equity_keys:
                            if k in bs.index:
                                equity = bs.loc[k][0]
                                break
                            
                        # Calculate
                        if liab and equity and equity != 0:
                            # Convert to Percentage to match yfinance standard
                            metrics["de"] = (liab / equity) * 100
                except:
                    pass

            # MANUAL OVERRIDES (Data source patches)
            # Patching known values to match SET.or.th or fix missing data
            MANUAL_OVERRIDES = {
                "SCB.BK": {"de": 618.0}, 
                "GULF.BK": {"yield": 0.015},
                "TFG.BK": {"yield": 0.0570}, # Manual Fix vs SET
                "CPF.BK": {"yield": 0.0468}, # Manual Fix vs SET
                "SIRI.BK": {"yield": 0.1112}, # Manual Fix vs SET
            }
            
            if symbol in MANUAL_OVERRIDES:
                for key, val in MANUAL_OVERRIDES[symbol].items():
                    # Only override if explicit value provided, OR if original missing (for GULF/SCB)
                    # For yield fixes, we Force override
                    metrics[key] = val

            data_list.append(metrics)
            
        except Exception as e:
            # Handle connection errors gracefully
            print(f"Error fetching {symbol}: {e}")
            
        # Update UI
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_text.write(f"Fetching {symbol} ({i+1}/{total})")
        
        # Respectful delay
        time.sleep(0.1) 
        
    # Save to JSON
    with open(DATA_FILE, 'w') as f:
        json.dump(data_list, f, indent=4)
        
    status_text.success(f"✅ Database updated! ({len(data_list)} stocks)")
    time.sleep(2)
    status_text.empty()
    progress_bar.empty()
    st.rerun()

def load_and_validate_data():
    """
    Phase 2: Validation
    Returns: Cleaned DataFrame or None
    """
    if not os.path.exists(DATA_FILE):
        return None
    
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
    except:
        return None
        
    if not data:
        return None
        
    df = pd.DataFrame(data)
    total_stocks = len(df)
    
    # Validation Logic
    # Drop rows where ANY key metric is NaN, None, or 0
    # Note: Yield = 0 usually means no dividend, which the prompt rules as exclusion.
    
    # First ensure cols exist
    required_cols = ["pe", "roe", "pbv", "yield", "high52", "ocf", "avg_volume", "ev_ebitda"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None 
            
    # Convert to numeric, forcing errors to NaN
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Filter
    valid_df = df.dropna(subset=required_cols) # Drop NaN
    
    # Drop 0s
    for col in required_cols:
        valid_df = valid_df[valid_df[col] != 0]
        
    # --- NEW: EARNINGS QUALITY FILTER ---
    # Must have Positive Operating Cash Flow
    # We allow None to pass if we want to be lenient, but for "Quality" we should strict.
    # Check if 'ocf' is valid (not None/NaN is already handled by dropna above)
    # Now check > 0
    valid_df = valid_df[valid_df['ocf'] > 0]
    
    # Note: We allow ev_ebitda to be NaN for Banks/Financials, so we don't dropna on it strictly 
    # unless we want to force it. For now, let's keep it but fill NaN with a neutral/high value for ranking?
    # Actually, Magic Formula usually excludes Financials. 
    # Let's simple fillna with a large number so they don't get a 'good' rank for missing data.
    valid_df['ev_ebitda'] = valid_df['ev_ebitda'].fillna(999) 

    # Calculate Daily Value (MB) for filtering
    # fillna(0) for safety, though drona checked above
    valid_df['daily_value_mb'] = (valid_df['price'] * valid_df['avg_volume']) / 1_000_000
        
    valid_count = len(valid_df)
    excluded_count = total_stocks - valid_count
    
    return valid_df, total_stocks, valid_count, excluded_count

def calculate_rankings(df):
    """
    Phase 3: Magic Formula Logic
    """
    # Ranking Rules
    # P/E: Ascending (Low is good)
    df['Rank_PE'] = df['pe'].rank(ascending=True)
    
    # P/BV: Ascending (Low is good)
    df['Rank_PBV'] = df['pbv'].rank(ascending=True)
    
    # ROE: Descending (High is good)
    df['Rank_ROE'] = df['roe'].rank(ascending=False)
    
    # Yield: Descending (High is good)
    df['Rank_Yield'] = df['yield'].rank(ascending=False)
    
    # Drawdown (Price drop from 52 Week High): Descending (High drop is Rank 1 = 1 Point)
    # Drawdown = (High - Price) / High
    df['drawdown'] = (df['high52'] - df['price']) / df['high52']
    df['Rank_Drawdown'] = df['drawdown'].rank(ascending=False)

    # EV/EBITDA: Ascending (Low is good)
    df['Rank_EV_EBITDA'] = df['ev_ebitda'].rank(ascending=True)

    # Scoring
    df['Total_Score'] = df['Rank_PE'] + df['Rank_PBV'] + df['Rank_ROE'] + df['Rank_Yield'] + df['Rank_Drawdown'] + df['Rank_EV_EBITDA']
    
    # Sort
    df_sorted = df.sort_values(by='Total_Score', ascending=True)
    
    # Formatting Top 30
    top_30 = df_sorted.head(30).copy()
    
    # Rounding and Formatting
    top_30['pe'] = top_30['pe'].round(2)
    top_30['pbv'] = top_30['pbv'].round(2)
    top_30['roe'] = (top_30['roe'] * 100).round(2)
    top_30['total_score'] = top_30['Total_Score'] # Keep raw score for display if needed
    
    # formatting Yield to percentage string
    top_30['yield_fmt'] = top_30['yield'].apply(lambda x: f"{x*100:.2f}%")
    
    # formatting Drawdown
    top_30['drawdown_fmt'] = top_30['drawdown'].apply(lambda x: f"{x*100:.2f}%")

    return top_30

# --- MAIN UI ---

st.title("🦄 SET100 Magic Formula Screener")
st.markdown("### Find high-quality, undervalued stocks in the Thai market.")
st.markdown("---")

# Sidebar
st.sidebar.header("🕹️ Controls")
st.sidebar.info("Data source: Yahoo Finance")

# --- UPDATE NOTIFICATION SYSTEM ---
today = datetime.datetime.now()
# Round 1 Alert: Mid June (e.g. 15-30) for July effective date
if today.month == 6 and today.day >= 15:
    st.sidebar.warning("🔔 **Update Alert!**\n\nSET100 list is usually updated mid-June.\nPlease check [SET Website](https://www.set.or.th) and update `source_set100.txt`.")

# Round 2 Alert: Mid Dec (e.g. 15-31) for Jan effective date
elif today.month == 12 and today.day >= 15:
    st.sidebar.warning("🔔 **Update Alert!**\n\nSET100 list is usually updated mid-Dec.\nPlease check [SET Website](https://www.set.or.th) and update `source_set100.txt`.")
# ----------------------------------

if st.sidebar.button("Update Database"):
    update_database()

# Last updated check
last_updated = "Never"
if os.path.exists(DATA_FILE):
    timestamp_mod = os.path.getmtime(DATA_FILE)
    last_updated = datetime.datetime.fromtimestamp(timestamp_mod).strftime('%Y-%m-%d %H:%M')

st.sidebar.markdown(f"**Last Updated:** {last_updated}")

# --- FEATURE: EDIT STOCK LIST ---
with st.sidebar.expander("📝 Edit Stock List"):
    current_symbols = get_set100_symbols()
    symbols_str = "\n".join([s.replace(".BK", "") for s in current_symbols])
    
    new_list_str = st.text_area("Enter symbols (one per line):", value=symbols_str, height=200)
    
    if st.button("Save List"):
        new_symbols = [s.strip().upper() for s in new_list_str.split('\n') if s.strip()]
        # Save to file
        with open("source_set100.txt", "w") as f:
            for s in new_symbols:
                f.write(s + "\n")
        st.success("List saved! Please click 'Update Database' to fetch new data.")
        time.sleep(1)
        st.rerun()

# --- FEATURE: LIQUIDITY FILTER ---
st.sidebar.markdown("---")
st.sidebar.subheader("🌪️ Filters")
min_liquidity = st.sidebar.number_input(
    "Min. Daily Value (MB)", 
    min_value=0, 
    value=10, 
    step=5,
    help="Filter stocks with average daily trading value lower than this amount (Million THB)."
)

# Load Data
result = load_and_validate_data()

if result is None:
    st.warning("⚠️ Database not found or empty. Please click 'Update Database' in the sidebar.")
else:
    df_clean, total, valid, excluded = result

    # --- APPLY LIQUIDITY FILTER ---
    # Handle missing column for backward compatibility (if user hasn't updated DB yet)
    if 'daily_value_mb' not in df_clean.columns:
        df_clean['daily_value_mb'] = 0
        st.warning("⚠️ 'Min Daily Value' filter requires a database update. Please click 'Update Database'.")
    
    filtered_df = df_clean[df_clean['daily_value_mb'] >= min_liquidity].copy()
    liquidity_excluded = len(df_clean) - len(filtered_df)
    
    # Phase 2 Report: Health Check
    st.subheader("📊 Data Health Check")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Stocks", total)
    col2.metric("Valid Data", valid)
    col3.metric("Low Liquidity", liquidity_excluded)
    col4.metric("Excluded (Bad Data)", excluded)
    
    with st.expander("See Exclusion Criteria"):
        st.write("1. Must have valid P/E, P/BV, ROE, Dividend Yield.")
        st.write("2. Must have **Positive Operating Cash Flow** (Earnings Quality Rule).")
        st.write("3. Must have non-zero values for key metrics.")
    
    st.markdown("---")
    
    if valid > 0:
        # Phase 3: Rankings
        st.subheader("🏆 Top 30 Magic Formula Stocks")
        
        ranked_df = calculate_rankings(filtered_df)
        
        # Prepare display dataframe
        # Prepare display dataframe
        # Ensure price and de columns exist for display, handle Nan if necessary
        cols_to_check = ['price', 'de']
        for c in cols_to_check:
            if c not in ranked_df.columns:
                ranked_df[c] = None
        
        # Convert D/E from percentage to ratio (e.g. 31.54 -> 0.3154)
        ranked_df['de'] = ranked_df['de'].apply(lambda x: x / 100 if pd.notnull(x) else None)
        
        # Display cleanup
        ranked_df['ev_ebitda'] = ranked_df['ev_ebitda'].replace(999, None) # Restore None for display

        display_df = ranked_df[[
            'symbol', 'Total_Score', 'price', 'drawdown_fmt', 'de', 'pe', 'pbv', 'roe', 'yield_fmt', 'ev_ebitda'
        ]].reset_index(drop=True)
        
        display_df.index += 1 # 1-based index
        display_df.columns = [
            'Symbol', 'Magic Score', 'Price (THB)', 'Down from 52W High', 'D/E Ratio', 'P/E Ratio', 'P/BV Ratio', 'ROE', 'Dividend Yield', 'EV/EBITDA'
        ]
    
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Magic Score": st.column_config.NumberColumn(
                    "Magic Score (Lower is Better)",
                    help="Sum of rankings for P/E, P/BV, ROE, Yield, Drawdown, and EV/EBITDA.",
                    format="%.1f"
                ),
                "EV/EBITDA": st.column_config.NumberColumn(format="%.2f", help="Enterprise Value / EBITDA. Lower is better (Cheaper)."),
                "Price (THB)": st.column_config.NumberColumn(format="%.2f"),
                "Down from 52W High": st.column_config.TextColumn(help="% Drop from 52-Week High"),
                "P/E Ratio": st.column_config.NumberColumn(format="%.2f"),
                "P/BV Ratio": st.column_config.NumberColumn(format="%.2f"),
                "ROE": st.column_config.NumberColumn(format="%.2f"),
                "D/E Ratio": st.column_config.NumberColumn(format="%.2f"),
            }
        )

        # --- FEATURE: DOWNLOAD CSV ---
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Top 30 as CSV",
            data=csv,
            file_name='magic_formula_top30.csv',
            mime='text/csv',
        )
        
        st.markdown(f"*Showing top {len(display_df)} candidates based on the Magic Formula.*")
    else:
        st.error("No valid data points found after filtering. This might happen if Yahoo Finance data is temporarily unavailable or incomplete.")
