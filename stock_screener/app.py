import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import datetime
import time
import altair as alt

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

def calculate_graham_number(eps, bvps):
    """
    Classic Benjamin Graham Formula (Fair Value)
    Valuation = Sqrt(22.5 * EPS * BVPS)
    """
    if eps is None or bvps is None or eps <= 0 or bvps <= 0:
        return None
    val = (22.5 * eps * bvps) ** 0.5
    return val

def calculate_rsi(series, period=14):
    """
    Relative Strength Index (RSI) Calculation
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Use Wilder's Smoothing Strategy (alpha=1/n)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
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
    Phase 1: ETL Process (Now with RSI!)
    """
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    symbols = get_set100_symbols()
    total = len(symbols)
    data_list = []
    
    status_text.write("🚀 Starting Data Extraction (Fundamental + Technical)...")
    
    for i, symbol in enumerate(symbols):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract metrics (using .get for safety)
            # Depending on yfinance version, keys might vary, but these are standard
            metrics = {
                "symbol": symbol,
                "sector": info.get("sector"),   # New: Industry/Sector
                "pe": info.get("trailingPE"),
                "roe": info.get("returnOnEquity"),
                "pbv": info.get("priceToBook"),
                "yield": info.get("dividendYield"),
                "price": info.get("currentPrice"),
                "de": info.get("debtToEquity"),
                "fcf": info.get("freeCashflow"),
                "shares": info.get("sharesOutstanding"),
                "eps": info.get("trailingEps"),   # For Graham Formula
                "bvps": info.get("bookValue"),    # For Graham Formula
                "growth": info.get("earningsGrowth"),
                "high52": info.get("fiftyTwoWeekHigh"),
                "ocf": info.get("operatingCashflow"),
                "avg_volume": info.get("averageVolume"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
                "total_assets": info.get("totalAssets"),        # F-Score
                "return_on_assets": info.get("returnOnAssets"), # F-Score
                "gross_margins": info.get("grossMargins"),      # F-Score
                "total_debt": info.get("totalDebt"),            # F-Score
                "current_ratio": info.get("currentRatio"),      # F-Score (Liquidity proxy)
                "revenue_growth": info.get("revenueGrowth"),    # F-Score (Efficiency proxy)
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # --- FALLBACK LOGIC ---
            
            # 1. P/E Fallback
            if metrics["pe"] is None:
                if info.get("forwardPE"):
                    metrics["pe"] = info.get("forwardPE")
                elif info.get("trailingEps") and info.get("trailingEps") > 0 and metrics["price"]:
                    metrics["pe"] = metrics["price"] / info.get("trailingEps")

            # 2. Yield Fallback
            try:
                divs = ticker.dividends
                if not divs.empty:
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
            
            if metrics["yield"] is None:
                div_rate = info.get("dividendRate")
                if div_rate and metrics["price"] and metrics["price"] > 0:
                     metrics["yield"] = div_rate / metrics["price"]

            # 3. D/E Fallback (Balance Sheet)
            if metrics["de"] is None:
                try:
                    bs = ticker.balance_sheet
                    if not bs.empty:
                        liab = None
                        target_liab_keys = ['Total Liabilities Net Minority Interest', 'Total Liabilities']
                        for k in target_liab_keys:
                            if k in bs.index:
                                liab = bs.loc[k].iloc[0]
                                break
                        equity = None
                        target_equity_keys = ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity']
                        for k in target_equity_keys:
                            if k in bs.index:
                                equity = bs.loc[k].iloc[0]
                                break
                        if liab and equity and equity != 0:
                            metrics["de"] = (liab / equity) * 100
                except:
                    pass

            # MANUAL OVERRIDES
            MANUAL_OVERRIDES = {
                "SCB.BK": {"de": 618.0}, 
                "GULF.BK": {"yield": 0.015},
                "TFG.BK": {"yield": 0.0570},
                "CPF.BK": {"yield": 0.0468},
                "SIRI.BK": {"yield": 0.1112},
            }
            if symbol in MANUAL_OVERRIDES:
                for key, val in MANUAL_OVERRIDES[symbol].items():
                    metrics[key] = val

            data_list.append(metrics)
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_text.write(f"Fetching {symbol} ({i+1}/{total})")
        
        time.sleep(0.1) 
        
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
    # Validation Logic: strict only on CORE Magic Formula params
    core_cols = ["pe", "roe", "pbv", "yield", "high52", "ocf"]
    secondary_cols = ["avg_volume", "ev_ebitda", "return_on_assets", "gross_margins", "eps", "bvps", "sector", "total_assets", "total_debt", "current_ratio", "revenue_growth"]
    
    all_cols = core_cols + secondary_cols
    for col in all_cols:
        if col not in df.columns:
            df[col] = None 
            
    # Convert to numeric, forcing errors to NaN (Only for numeric cols)
    numeric_cols = [c for c in all_cols if c != "sector"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Filter: Drop only if CORE metrics are missing
    valid_df = df.dropna(subset=core_cols) 
    
    # Drop 0s in core metrics (some can be effectively 0 but let's stick to safe Magic Formula rules)
    for col in ["pe", "pbv"]: # Yield can be 0, ROE can be 0 or negative
         valid_df = valid_df[valid_df[col] != 0]

    # --- NEW: EARNINGS QUALITY FILTER ---
    # Must have Positive Operating Cash Flow
    # We allow None to pass if we want to be lenient, but for "Quality" we should strict.
    # Check if 'ocf' is valid (not None/NaN is already handled by dropna above)
    # Now check > 0
    valid_df = valid_df[valid_df['ocf'] > 0]
    
    # Calculate Piotroski F-Score (Simplified with available data)
    # We use a simplified version because full history is expensive to fetch in loop
    # 1. ROA > 0
    # 2. CFO > 0
    # 3. CFO > ROA (Quality of Earnings)
    # 4. Long Term Debt < Prior Year (Proxy: Debt/Equity check low or decreasing? Hard. Skip or use simple check)
    # 5. Current Ratio > Prior Year (Proxy: Safe Current Ratio > 1.5)
    # 6. No New Shares (Skip)
    # 7. Gross Margin > Prior (Proxy: Positive Gross Margin)
    # 8. Asset Turnover > Prior (Proxy: Positive Revenue Growth)
    
    # Let's implement a 'Modified F-Score' based on Snapshot Data
    def calculate_modified_f_score(row):
        score = 0
        try:
            # 1. Positive ROA
            if row['return_on_assets'] and row['return_on_assets'] > 0: score += 1
            
            # 2. Positive OCF
            if row['ocf'] and row['ocf'] > 0: score += 1
            
            # 3. Quality of earnings (CFO > Net Income). 
            # Note: We don't have Net Income explicit, but can imply. 
            # Simplified: CFO / Assets > ROA
            if row['total_assets'] and row['total_assets'] > 0:
                cfo_roa = row['ocf'] / row['total_assets']
                if row['return_on_assets'] and cfo_roa > row['return_on_assets']: score += 1
            
            # 4. Low Leverage (D/E < 1 is safe benchmark for general)
            if row['de'] and row['de'] < 100: score += 1 # D/E is percent in our data
            
            # 5. Liquidity (Current Ratio > 1)
            if row['current_ratio'] and row['current_ratio'] > 1.0: score += 1
            
            # 6. Improving Efficiency (Revenue Growth > 0)
            if row['revenue_growth'] and row['revenue_growth'] > 0: score += 1
            
            # 7. Positive Gross Margin
            if row['gross_margins'] and row['gross_margins'] > 0: score += 1
            
            # 8. Yield (Pays Dividend)
            if row['yield'] and row['yield'] > 0: score += 1
            
            # 9. Low Valuation Premium (P/E < 15)
            if row['pe'] and row['pe'] < 15: score += 1
            
        except:
            pass
        return score

    valid_df['f_score'] = valid_df.apply(calculate_modified_f_score, axis=1)

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

    # --- INTRINSIC VALUE & MOS (New Feature) ---
    top_30['fair_value'] = top_30.apply(lambda row: calculate_graham_number(row['eps'], row['bvps']), axis=1)
    
    def calc_mos(row):
        if row['fair_value'] and row['price'] and row['fair_value'] > 0:
            return (row['fair_value'] - row['price']) / row['fair_value']
        return None

    top_30['mos'] = top_30.apply(calc_mos, axis=1)
    
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
    

    
    if valid > 0:
        # Phase 3: Rankings
        
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
        ranked_df['ev_ebitda'] = ranked_df['ev_ebitda'].replace(999, None) 
        
        # Enable Right-Alignment for Percentage Columns by keeping them numeric
        ranked_df['yield_pct'] = ranked_df['yield'] * 100
        ranked_df['drawdown_pct'] = ranked_df['drawdown'] * 100
        ranked_df['mos_pct'] = ranked_df['mos'].apply(lambda x: x * 100 if pd.notnull(x) else 0)

        # Map Sector to Thai
        sector_map = {
            "Energy": "พลังงาน",
            "Financial Services": "เงินทุน/หลักทรัพย์",
            "Real Estate": "อสังหาริมทรัพย์",
            "Industrials": "สินค้าอุตสาหกรรม",
            "Basic Materials": "วัสดุก่อสร้าง/เคมีภัณฑ์",
            "Consumer Cyclical": "สินค้าบริการ",
            "Consumer Defensive": "สินค้าอุปโภคบริโภค",
            "Healthcare": "การแพทย์",
            "Utilities": "สาธารณูปโภค",
            "Technology": "เทคโนโลยี",
            "Communication Services": "สื่อสาร",
            "None": "ไม่ระบุ"
        }
        ranked_df['sector_th'] = ranked_df['sector'].map(sector_map).fillna(ranked_df['sector']) # Use English if no map

        # --- LOGIC: Auto-Rating ---
        def get_rating(row):
            # 1. Financial Strength (F-Score >= 7 is Strong)
            fs = row['f_score'] if pd.notnull(row['f_score']) else 0
            strong_fin = fs >= 7
            weak_fin = fs <= 4
            
            # 2. Valuation (MOS > 0 is Undervalued)
            mos = row['mos'] if pd.notnull(row['mos']) else -1
            cheap = mos > 0
            very_cheap = mos > 0.20
            
            # 3. Risk (High Debt > 2.5)
            de_ratio = row['de'] if pd.notnull(row['de']) else 0
            hight_risk = de_ratio > 2.5
            
            if strong_fin and very_cheap and not hight_risk:
                return "⭐⭐⭐" # Top Pick
            elif (strong_fin and cheap) or (fs >= 6 and very_cheap):
                return "⭐⭐"     # Good
            elif weak_fin or hight_risk:
                return "⚠️"      # Caution
            else:
                return "😐"      # Neutral

        ranked_df['rating_icon'] = ranked_df.apply(get_rating, axis=1)

        # --- LOGIC: Quadrant Category ---
        def get_quadrant(row):
            pe = row['pe'] if pd.notnull(row['pe']) else 999
            roe = row['roe'] if pd.notnull(row['roe']) else 0
            
            # Benchmarks: PE=15, ROE=12
            if pe < 15 and roe >= 12:
                return "💎 ของดีราคาถูก"
            elif pe >= 15 and roe >= 12:
                return "⭐ แพงแต่ดี"
            elif pe < 15 and roe < 12:
                return "⚠️ ระวังกับดัก"
            else:
                return "❌ แพงและแย่"

        ranked_df['quadrant'] = ranked_df.apply(get_quadrant, axis=1)

        display_df = ranked_df[[
            'symbol', 'rating_icon', 'quadrant', 'sector_th', 'Total_Score', 'f_score', 'fair_value', 'mos_pct', 'price', 'drawdown_pct', 'de', 'pe', 'pbv', 'roe', 'ev_ebitda', 'yield_pct'
        ]].reset_index(drop=True)
        
        display_df.index += 1 
        display_df.columns = [
            'Symbol', 'Rating', 'Stock Category', 'Industry', 'Magic Score', 'F-Score', 'Graham Fair Value', 'M.O.S', 'Price (THB)', 'Down from 52W High', 'D/E Ratio', 'P/E Ratio', 'P/BV Ratio', 'ROE', 'EV/EBITDA', 'Dividend Yield'
        ]

        # --- GOD MODE: VISUALIZATION ---
        st.markdown("---")
        st.subheader("🔮 Magic Quadrant: แผนที่ขุมทรัพย์ 🗺️")
        st.info("💡 **วิธีดูง่ายๆ:** ผมแบ่งโซนไว้ให้แล้วครับ! มองหาหุ้นในโซน **'สีเขียว (ซ้ายบน)'** นั่นคือ **'ของดีราคาถูก'** ที่น่าลงทุนที่สุดครับ")
        
        # Prepare Chart Data (Filter outliers for better chart)
        chart_data = ranked_df[ranked_df['pe'] < 40].copy()
        chart_data['sector_th'] = chart_data['sector_th'].fillna('Unknown')
        
        # 1. Base Points
        base = alt.Chart(chart_data).encode(
            x=alt.X('pe', title='ความถูก (P/E Ratio) ← ยิ่งซ้ายยิ่งถูก', scale=alt.Scale(domain=[0, 30], clamp=True)),
            y=alt.Y('roe', title='คุณภาพ (ROE %) ↑ ยิ่งบนยิ่งเก่ง', scale=alt.Scale(domain=[0, 40], clamp=True))
        )

        points = base.mark_circle(size=140, opacity=0.9).encode(
            color=alt.Color('sector_th', legend=alt.Legend(title="กลุ่มอุตสาหกรรม")),
            tooltip=['symbol', 'price', 'pe', 'roe', 'f_score', 'rating_icon']
        )

        # 2. Quadrant Dividers (Benchmarks: P/E=15, ROE=12)
        h_line = alt.Chart(pd.DataFrame({'y': [12]})).mark_rule(strokeDash=[3, 3], color='gray', opacity=0.5).encode(y='y')
        v_line = alt.Chart(pd.DataFrame({'x': [15]})).mark_rule(strokeDash=[3, 3], color='gray', opacity=0.5).encode(x='x')

        # 3. Text Labels (The "Thinking" Part)
        labels_df = pd.DataFrame({
            'x': [7.5, 22.5, 7.5, 22.5],
            'y': [35, 35, 5, 5],
            'label': ['💎 ของดีราคาถูก (Buy!)', '⭐ แพงแต่ดี (Growth)', '⚠️ ระวังกับดัก (Value Trap)', '❌ แพงและแย่ (Avoid)'],
            'color': ['#27ae60', '#f39c12', '#e67e22', '#c0392b']
        })
        
        text_labels = alt.Chart(labels_df).mark_text(
            align='center', baseline='middle', fontSize=16, fontWeight='bold'
        ).encode(
            x='x', y='y', text='label', color=alt.Color('color', scale=None)
        )

        final_chart = (points + h_line + v_line + text_labels).interactive()
        
        st.altair_chart(final_chart, use_container_width=True)
        st.markdown("---")
    
        st.subheader("🏆 Top 30 Magic Formula Stocks")
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Symbol": st.column_config.TextColumn(width="small"),
                "Rating": st.column_config.TextColumn(width="small", help="⭐⭐⭐ = Strong F-Score & Undervalued"),
                "Stock Category": st.column_config.TextColumn(width="medium", help="Based on Magic Quadrant (P/E vs ROE)"),
                "Industry": st.column_config.TextColumn(width="medium"),
                "Magic Score": st.column_config.NumberColumn(format="%.1f", width="small"),
                "F-Score": st.column_config.ProgressColumn(min_value=0, max_value=9, format="%d", width="small"),
                "Graham Fair Value": st.column_config.NumberColumn(format="%.2f", width="small"),
                "M.O.S": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%", width="small"),
                "Price (THB)": st.column_config.NumberColumn(format="%.2f", width="small"),
                "Down from 52W High": st.column_config.NumberColumn(format="%.2f%%", width="small"),
                "D/E Ratio": st.column_config.NumberColumn(format="%.2f", width="small"),
                "P/E Ratio": st.column_config.NumberColumn(format="%.2f", width="small"),
                "P/BV Ratio": st.column_config.NumberColumn(format="%.2f", width="small"),
                "ROE": st.column_config.NumberColumn(format="%.2f", width="small"),
                "EV/EBITDA": st.column_config.NumberColumn(format="%.2f", width="small"),
                "Dividend Yield": st.column_config.NumberColumn(format="%.2f%%", width="small"),
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
        
        st.markdown("---")
        st.subheader("📖 คำอธิบาย (Glossary)")
        st.markdown("""
        *   **Magic Score:** คะแนนจัดอันดับรวมจากสูตร (ยิ่ง **น้อย** ยิ่งดี) เกิดจากการรวมอันดับของความถูก (P/E, P/BV, EV/EBITDA, Drawdown) และคุณภาพ (ROE, Yield)
        *   **F-Score:** คะแนนสุขภาพทางการเงิน (เต็ม 9) วัดจากกำไร, หนี้สิน, และประสิทธิภาพบริษัท **(7-9 = แข็งแกร่ง)**
        *   **Graham Fair Value:** ราคาที่เหมาะสมตามทฤษฎี Benjamin Graham (เน้นมูลค่าทางบัญชีและกำไรต่อหุ้น)
        *   **M.O.S (Margin of Safety):** ส่วนเผื่อความปลอดภัย บอกว่าราคาปัจจุบัน **ถูกกว่า** ราคาเหมาะสมอยู่กี่ % (ยิ่งเยอะยิ่งปลอดภัย)
        *   **P/E Ratio:** ราคาหุ้นเป็นกี่เท่าของกำไร (ยิ่งต่ำยิ่งดี) / **P/BV Ratio:** ราคาหุ้นเป็นกี่เท่าของมูลค่าทางบัญชี (ยิ่งต่ำยิ่งดี)
        *   **ROE:** บริษัททำกำไรได้กี่ % จากเงินลงทุนของผู้ถือหุ้น (ยิ่งสูงยิ่งดี) 
        *   **EV/EBITDA:** มูลค่ากิจการเทียบกับกำไรเงินสด (Cash Flow) เป็นค่าที่ขจัดเรื่องโครงสร้างหนี้และภาษีออกไป (ยิ่งต่ำยิ่งดี)
        *   **D/E Ratio:** หนี้สินต่อทุน (ยิ่งต่ำยิ่งปลอดภัย ไม่ควรเกิน 1-2 เท่า)
        *   **Down from 52W High:** ราคาลดลงจากจุดสูงสุดในรอบปีมาแล้วกี่ % (ใช้หาหุ้นที่เพิ่งลงมา)
        """)
    else:
        st.error("No valid data points found after filtering. This might happen if Yahoo Finance data is temporarily unavailable or incomplete.")

