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

def calculate_intrinsic_value(fcf, shares, growth_rate, discount_rate=0.12, terminal_growth=0.02, years=10):
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
    elif growth_rate > 0.05:
        g = 0.05 # Cap at 5% (Conservative for Thai Market)
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
                "growth": info.get("earningsGrowth"),
                "high52": info.get("fiftyTwoWeekHigh"),
                "ocf": info.get("operatingCashflow"),
                "evebitda": info.get("enterpriseToEbitda"),
                "volume": info.get("averageVolume"),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # --- FALLBACK LOGIC ---
            
            # 1. EV/EBITDA Fallback (Calculate if missing)
            if metrics["evebitda"] is None:
                ev = info.get("enterpriseValue")
                ebitda = info.get("ebitda")
                if ev and ebitda and ebitda > 0:
                     metrics["evebitda"] = ev / ebitda
                
                # --- HYBRID LOGIC FOR BANKS/FINANCE ---
                # Banks don't have EBITDA. If EV/EBITDA is still None, use P/E as proxy.
                if metrics["evebitda"] is None and metrics["pe"] is not None:
                    metrics["evebitda"] = metrics["pe"] # Use P/E score in the EV/EBITDA slot
                    metrics["is_pe_proxy"] = True # Flag for UI to show (P/E)
                else:
                    metrics["is_pe_proxy"] = False

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
    # Switched 'pe' to 'evebitda'
    # Need 'fcf', 'shares', 'growth', 'volume'
    required_cols = ["evebitda", "roe", "pbv", "yield", "high52", "ocf"]
    numeric_cols = required_cols + ["fcf", "shares", "growth", "price", "volume"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None 
            
    # Convert to numeric, forcing errors to NaN
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Filter
    valid_df = df.dropna(subset=required_cols) # Drop NaN
    
    # Drop 0s
    for col in required_cols:
        valid_df = valid_df[valid_df[col] != 0]
        
    # --- NEW: EARNINGS QUALITY FILTER ---
    # Must have Positive Operating Cash Flow
    # EXCEPTION: Banks/Insurance often have negative operating cashflow due to loan issuance mechanics
    # Ideally we check sector, but simple proxy is: if P/E Proxy was used, we might relax OCF rule
    # OR we just stick to OCF > 0.
    # Let's keep OCF > 0 but be aware BBL might fail it (-5B OCF).
    # If we want banks, we might need to relax this for them.
    # Let's try: Pass if OCF > 0 OR (is_pe_proxy is True [likely bank])
    
    # Ensure is_pe_proxy exists
    if 'is_pe_proxy' not in valid_df.columns:
        valid_df['is_pe_proxy'] = False
        
    valid_df = valid_df[ (valid_df['ocf'] > 0) | (valid_df['is_pe_proxy'] == True) ]
        
    valid_count = len(valid_df)
    excluded_count = total_stocks - valid_count
    
    return valid_df, total_stocks, valid_count, excluded_count

def calculate_rankings(df):
    """
    Phase 3: Magic Formula Logic
    """
    # Ranking Rules
    # Valuation: Ascending (Low is good) - Hybrid (EV/EBITDA or P/E)
    df['Rank_Valuation'] = df['evebitda'].rank(ascending=True)
    
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

    # Scoring
    df['Total_Score'] = df['Rank_Valuation'] + df['Rank_PBV'] + df['Rank_ROE'] + df['Rank_Yield'] + df['Rank_Drawdown']
    
    # Sort
    df_sorted = df.sort_values(by='Total_Score', ascending=True)
    
    # Formatting Top 30
    top_30 = df_sorted.head(30).copy()
    
    # Rounding and Formatting
    # Format Valuation column to show source
    def format_val(row):
        val = row['evebitda']
        if pd.isna(val): return "N/A"
        if row.get('is_pe_proxy', False):
            return f"{val:.2f} (P/E)"
        return f"{val:.2f}"

    top_30['val_fmt'] = top_30.apply(format_val, axis=1)
    
    top_30['pbv'] = top_30['pbv'].round(2)
    top_30['roe'] = (top_30['roe'] * 100).round(2)
    top_30['total_score'] = top_30['Total_Score'] # Keep raw score for display if needed
    
    # formatting Yield to percentage string
    top_30['yield_fmt'] = top_30['yield'].apply(lambda x: f"{x*100:.2f}%")
    
    # formatting Drawdown
    top_30['drawdown_fmt'] = top_30['drawdown'].apply(lambda x: f"{x*100:.2f}%")
    
    # --- NEW: CALCULATE FAIR VALUE & MOS ---
    # We do this calculation only for the Top 30 to save processing (though it's fast)
    top_30['Fair_Value'] = top_30.apply(
        lambda row: calculate_intrinsic_value(row['fcf'], row['shares'], row['growth']), axis=1
    )
    
    # Calculate Margin of Safety (MOS) / Upside
    # Upside = (Fair - Price) / Price
    top_30['mos'] = top_30.apply(
        lambda row: ((row['Fair_Value'] - row['price']) / row['price']) if row['Fair_Value'] and row['price'] else None, axis=1
    )
    
    # Format MOS
    def format_mos(x):
        if x is None:
            return "N/A"
        return f"{x*100:.1f}%"
        
    top_30['mos_fmt'] = top_30['mos'].apply(format_mos)

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
st.sidebar.markdown("---")

# --- STOCK LIST EDITOR ---
with st.sidebar.expander("📝 Edit Stock List"):
    if os.path.exists("source_set100.txt"):
        with open("source_set100.txt", "r") as f:
            current_list = f.read()
    else:
        current_list = ""
        
    new_list = st.text_area("SET100 Symbols (One per line)", current_list, height=150)
    
    if st.button("Save List"):
        with open("source_set100.txt", "w") as f:
            f.write(new_list)
        st.success("List saved!")
        time.sleep(1)
        st.rerun()

# Load Data
result = load_and_validate_data()

if result is None:
    st.warning("⚠️ Database not found or empty. Please click 'Update Database' in the sidebar.")
else:
    df_clean, total, valid, excluded = result
    
    # Phase 2 Report: Health Check
    st.subheader("📊 Data Health Check")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Stocks", total)
    col2.metric("Complete Data", valid)
    col3.metric("Excluded (Incomplete/Zero)", excluded)
    
    with st.expander("See Exclusion Criteria"):
        st.write("1. Must have valid EV/EBITDA, P/BV, ROE, Dividend Yield.")
        st.write("2. Must have **Positive Operating Cash Flow** (Earnings Quality Rule).")
        st.write("3. Must have non-zero values for key metrics.")
    
    # --- LIQUIDITY FILTER ---
    st.sidebar.markdown("---")
    st.sidebar.header("💧 Filters")
    min_vol_thb = st.sidebar.slider(
        "Min Daily Value (THB Million)", 
        min_value=0, max_value=50, value=3, step=1,
        help="Filter out stocks with average daily trading value below this amount."
    ) * 1_000_000
    
    # Apply Filter
    # Avg Value = Avg Volume * Current Price
    # We construct a temp column for this check
    # Ensure volume and price are present
    if 'volume' in df_clean.columns and 'price' in df_clean.columns:
        df_clean['avg_value_thb'] = df_clean['volume'] * df_clean['price']
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['avg_value_thb'] >= min_vol_thb]
        filtered_out = initial_count - len(df_clean)
        if filtered_out > 0:
            st.sidebar.caption(f"Filtered out {filtered_out} low liquidity stocks.")
    
    st.markdown("---")
    
    if valid > 0:
        # Phase 3: Rankings
        st.subheader("🏆 Top 30 Magic Formula Stocks")
        
        ranked_df = calculate_rankings(df_clean)
        
        # Prepare display dataframe
        # Prepare display dataframe
        # Ensure price and de columns exist for display, handle Nan if necessary
        cols_to_check = ['price', 'de']
        for c in cols_to_check:
            if c not in ranked_df.columns:
                ranked_df[c] = None
        
        # Convert D/E from percentage to ratio (e.g. 31.54 -> 0.3154)
        ranked_df['de'] = ranked_df['de'].apply(lambda x: x / 100 if pd.notnull(x) else None)

        display_df = ranked_df[[
            'symbol', 'Total_Score', 'price', 'drawdown_fmt', 'val_fmt', 'pbv', 'roe', 'yield_fmt'
        ]].reset_index(drop=True)
        
        display_df.index += 1 # 1-based index
        display_df.columns = [
            'Symbol', 'Magic Score', 'Price (THB)', 'Down from High', 'Valuation (EV/EBITDA)', 'P/BV', 'ROE', 'Yield'
        ]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Magic Score": st.column_config.NumberColumn(
                    "Magic Score (Lower is Better)",
                    help="Sum of rankings for EV/EBITDA, P/BV, ROE, Yield, and Drawdown.",
                    format="%.1f"
                ),
                "Price (THB)": st.column_config.NumberColumn(format="%.2f"),
                "Down from High": st.column_config.TextColumn(help="% Drop from 52-Week High"),
                "Valuation (EV/EBITDA)": st.column_config.TextColumn(help="Mainly EV/EBITDA. If marked (P/E), it uses P/E Ratio (for Banks/Finance)."),
                "P/BV": st.column_config.NumberColumn(format="%.2f"),
                "ROE": st.column_config.NumberColumn(format="%.2f"),
            }
        )
        
        st.markdown(f"*Showing top {len(display_df)} candidates based on the Magic Formula.*")
        
        # CSV Download
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Top 30 as CSV",
            data=csv,
            file_name=f'magic_formula_top30_{datetime.date.today()}.csv',
            mime='text/csv',
        )
    else:
        st.error("No valid data points found after filtering. This might happen if Yahoo Finance data is temporarily unavailable or incomplete.")
