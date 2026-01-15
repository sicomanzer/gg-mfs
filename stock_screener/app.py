import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import json
import os
import random
import datetime
import time
import altair as alt


# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "set100_db.json")
SOURCE_FILE = os.path.join(BASE_DIR, "source_set100.txt")

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
    if os.path.exists(SOURCE_FILE):
        with open(SOURCE_FILE, "r") as f:
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
    
    # Create a spoofed session to bypass basic IP blocks
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })

    symbols = get_set100_symbols()
    
    # Check if empty (e.g. bad user edit)
    if not symbols:
        st.sidebar.warning("⚠️ Stock list is blank! Using fallback list.")
        symbols = [f"{s}.BK" for s in FALLBACK_SET100]

    total = len(symbols)
    data_list = []
    errors = [] # Track errors
    
    status_text.write("🚀 Starting Update... (Please wait)")
    
    for i, symbol in enumerate(symbols):
        try:
            # Try with Spoofed Session first (Best for Rate Limit)
            try:
                ticker = yf.Ticker(symbol, session=session)
                # Test fetch
                info = ticker.info 
                if info is None: raise ValueError("Info is None")
            except Exception as session_err:
                # Fallback to standard request if session fails (e.g. yfinance version issue)
                # print(f"Session failed for {symbol}: {session_err}, trying standard...")
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
                "roic": info.get("returnOnInvestedCapital"),    # Magic Formula Original
                "beta": info.get("beta"),                       # For WACC
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

            # 3. D/E Fallback (Balance Sheet) & ROIC Fallback
            if metrics["de"] is None or metrics["roic"] is None:
                try:
                    bs = ticker.balance_sheet
                    if not bs.empty:
                        # --- D/E ---
                        if metrics["de"] is None:
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
                        
                        # --- ROIC Calculation (EBIT / Invested Capital) ---
                        if metrics["roic"] is None:
                            try:
                                # 1. Get EBIT
                                financials = ticker.financials
                                ebit = None
                                if not financials.empty and 'Ebit' in financials.index:
                                    ebit = financials.loc['Ebit'].iloc[0]
                                elif not financials.empty and 'Net Income' in financials.index:
                                    # Very Rough Proxy: Net Income + Interest + Tax
                                    ebit = financials.loc['Net Income'].iloc[0] # Too rough, maybe just skip
                                
                                # 2. Get Invested Capital (Equity + Debt - Cash)
                                has_equity = 'Stockholders Equity' in bs.index or 'Total Stockholder Equity' in bs.index or 'Common Stock Equity' in bs.index
                                
                                if ebit and has_equity:
                                    # Equity
                                    for k in ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity']:
                                        if k in bs.index:
                                            eq_val = bs.loc[k].iloc[0]
                                            break
                                    
                                    # Debt
                                    debt_val = 0
                                    for k in ['Total Debt', 'Long Term Debt']: # Simplified
                                        if k in bs.index:
                                            debt_val = bs.loc[k].iloc[0]
                                            break
                                    
                                    # Cash
                                    cash_val = 0
                                    for k in ['Cash And Cash Equivalents', 'Cash']:
                                        if k in bs.index:
                                            cash_val = bs.loc[k].iloc[0]
                                            break
                                    
                                    invested_capital = eq_val + debt_val - cash_val
                                    
                                    if invested_capital and invested_capital > 0:
                                        metrics["roic"] = ebit / invested_capital
                            except:
                                pass
                except:
                    pass
            
            # Final Fallback for ROIC: Use ROA as proxy if still None
            if metrics["roic"] is None and metrics["return_on_assets"] is not None:
                 metrics["roic"] = metrics["return_on_assets"]

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
            # print(f"Error fetching {symbol}: {e}")
            errors.append(f"{symbol}: {str(e)}")
            
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_text.write(f"Fetching {symbol} ({i+1}/{total})")
        
        # Rate Limit Mitigation for Render/Cloud
        # 1. Random sleep 1-3 seconds
        sleep_time = random.uniform(1.0, 3.0)
        time.sleep(sleep_time)
        
        # 2. Cool down every 10 stocks
        if (i + 1) % 10 == 0:
            time.sleep(5) 
            
    # --- SAFETY CHECK: DON'T SAVE IF EMPTY ---
    if not data_list:
        st.sidebar.error("❌ Update Failed: No data fetched!")
        st.sidebar.warning("Possible reasons: Connection blocked (429), Empty list, or Yahoo API change.")
        if errors:
            with st.sidebar.expander("Show Errors"):
                st.write(errors)
        return # EXIT WITHOUT SAVING

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
    secondary_cols = ["avg_volume", "ev_ebitda", "return_on_assets", "gross_margins", "eps", "bvps", "sector", "total_assets", "total_debt", "current_ratio", "revenue_growth", "roic", "beta"]
    
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

def calculate_rankings(df, use_roic=False):
    """
    Phase 3: Magic Formula Logic
    """
    # Ranking Rules
    # P/E: Ascending (Low is good)
    df['Rank_PE'] = df['pe'].rank(ascending=True)
    
    # P/BV: Ascending (Low is good)
    df['Rank_PBV'] = df['pbv'].rank(ascending=True)
    
    # Quality Metric: ROE or ROIC
    # Descending (High is good)
    if use_roic:
        # Fill missing ROIC with -999 to rank them last
        df['roic_filled'] = df['roic'].fillna(-999)
        df['Rank_Quality'] = df['roic_filled'].rank(ascending=False)
    else:
        df['Rank_Quality'] = df['roe'].rank(ascending=False)
    
    # Yield: Descending (High is good)
    df['Rank_Yield'] = df['yield'].rank(ascending=False)
    
    # Drawdown (Price drop from 52 Week High): Descending (High drop is Rank 1 = 1 Point)
    # Drawdown = (High - Price) / High
    df['drawdown'] = (df['high52'] - df['price']) / df['high52']
    df['Rank_Drawdown'] = df['drawdown'].rank(ascending=False)

    # EV/EBITDA: Ascending (Low is good)
    df['Rank_EV_EBITDA'] = df['ev_ebitda'].rank(ascending=True)

    # Scoring
    df['Total_Score'] = df['Rank_PE'] + df['Rank_PBV'] + df['Rank_Quality'] + df['Rank_Yield'] + df['Rank_Drawdown'] + df['Rank_EV_EBITDA']
    
    # Sort
    df_sorted = df.sort_values(by='Total_Score', ascending=True)
    
    # Formatting Top 30
    top_30 = df_sorted.head(30).copy()
    
    # Rounding and Formatting
    top_30['pe'] = top_30['pe'].round(2)
    top_30['pbv'] = top_30['pbv'].round(2)
    top_30['roe'] = (top_30['roe'] * 100).round(2)
    if 'roic' in top_30.columns:
        top_30['roic'] = (top_30['roic'] * 100).round(2)
        
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
        with open(SOURCE_FILE, "w") as f:
            for s in new_symbols:
                f.write(s + "\n")
        st.success("List saved! Please click 'Update Database' to fetch new data.")
        time.sleep(1)
        st.rerun()

# --- FEATURE: LIQUIDITY FILTER ---
st.sidebar.markdown("---")
st.sidebar.subheader("🌪️ Filters & Config")
quality_selector = st.sidebar.radio(
    "Magic Formula Quality Metric:",
    ["ROE (Return on Equity)", "ROIC (Return on Invested Capital)"],
    help="Original Magic Formula uses ROIC. Modified version uses ROE for broader data availability."
)
use_roic = "ROIC" in quality_selector

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
        
        ranked_df = calculate_rankings(filtered_df, use_roic=use_roic)
        
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

        if 'roic' not in ranked_df.columns: ranked_df['roic'] = None
        
        display_df = ranked_df[[
            'symbol', 'rating_icon', 'quadrant', 'sector_th', 'Total_Score', 'f_score', 'fair_value', 'mos_pct', 'price', 'drawdown_pct', 'de', 'pe', 'pbv', 'roe', 'roic', 'ev_ebitda', 'yield_pct'
        ]].reset_index(drop=True)
        
        display_df.index += 1 
        display_df.columns = [
            'Symbol', 'Rating', 'Stock Category', 'Industry', 'Magic Score', 'F-Score', 'Graham Fair Value', 'M.O.S', 'Price (THB)', 'Down from 52W High', 'D/E Ratio', 'P/E Ratio', 'P/BV Ratio', 'ROE', 'ROIC', 'EV/EBITDA', 'Dividend Yield'
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
                "Stock Category": st.column_config.TextColumn(width=150, help="Based on Magic Quadrant (P/E vs ROE)"),
                "Industry": st.column_config.TextColumn(width=150),
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
                "ROIC": st.column_config.NumberColumn(format="%.2f", width="small"),
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

        # --- FEATURE: STOCK DEEP DIVE ---
        st.markdown("---")
        st.subheader("🕵️‍♂️ Stock Deep Dive (เจาะลึกหุ้นรายตัว)")
        
        col_sel1, col_sel2 = st.columns([1, 2])
        with col_sel1:
            stock_list = ranked_df['symbol'].tolist()
            selected_stock = st.selectbox("เลือกหุ้นเพื่อดูรายละเอียด:", stock_list)
        
        if selected_stock:
            # Get Row
            row = ranked_df[ranked_df['symbol'] == selected_stock].iloc[0]
            
            # --- AI INVESTMENT CALL LOGIC ---
            # 1. Valuation Check (Based on Graham MOS)
            mos_pct = row['mos_pct']
            
            # 2. Quality Check (F-Score)
            f_score = row.get('f_score', 0)
            
            # 3. Decision Matrix
            rec_status = "HOLD"
            rec_color = "gray"
            rec_reason = "พื้นฐานปานกลาง ราคาเหมาะสม"
            
            if mos_pct >= 40: # Deeply Undervalued
                if f_score >= 7:
                    rec_status = "STRONG BUY (ซื้อสะสม)"
                    rec_color = "green"
                    rec_reason = "💎 หุ้นห่านทองคำ: ราคาถูกมาก (MOS > 40%) และพื้นฐานแกร่ง (F-Score สูง)"
                elif f_score >= 5:
                    rec_status = "BUY (ทยอยซื้อ)"
                    rec_color = "lightgreen"
                    rec_reason = "✅ หุ้นดีราคาถูก: มีส่วนลดเยอะ (MOS > 40%) พื้นฐานผ่านเกณฑ์"
                else:
                    rec_status = "WAIT (รอคอย)"
                    rec_color = "orange"
                    rec_reason = "⚠️ กับดักมูลค่า?: ราคาถูกแต่พื้นฐานอ่อนแอ (F-Score ต่ำ) // โปรดเช็ดงบ"
                    
            elif mos_pct >= 10: # Undervalued
                if f_score >= 6:
                    rec_status = "BUY (ซื้อ)"
                    rec_color = "lightgreen"
                    rec_reason = "✅ หุ้นดีมีส่วนลด: ราคาต่ำกว่ามูลค่า (MOS > 10%) พื้นฐานดี"
                else:
                    rec_status = "HOLD (ถือ/รอ)"
                    rec_color = "yellow"
                    rec_reason = "🟡 ราคาเริ่มน่าสนใจ แต่พื้นฐานยังไม่แน่นปึก"
                    
            elif mos_pct >= -10: # Fair Value (approx)
                if f_score >= 7:
                    rec_status = "HOLD (ถือ)"
                    rec_color = "blue"
                    rec_reason = "🛡️ หุ้นแกร่ง: ราคายุติธรรม เน้นถือรับปันผล/Growth"
                else:
                    rec_status = "WAIT (รอ)"
                    rec_color = "orange"
                    rec_reason = "🟠 ราคาแฟร์แต่พื้นฐานไม่เด่น รอจังหวะย่อตัว"
            
            else: # Overvalued
                if f_score >= 8:
                    rec_status = "HOLD (ถือรอขาย)"
                    rec_color = "blue"
                    rec_reason = "💎 ของดีราคาแพง: พื้นฐานเทพ แต่อาจจะแพงไปนิด ถือลุ้น Growth ต่อได้"
                else:
                    rec_status = "SELL (พิจารณาขาย)"
                    rec_color = "red"
                    rec_reason = "❌ แพงเกินพื้นฐาน: ราคาเกินมูลค่าและคุณภาพไม่รองรับ"

            # Display Recommendation Banner
            st.markdown(f"""
                <div style="
                    background-color: {rec_color}; 
                    padding: 15px; 
                    border-radius: 10px; 
                    color: {'white' if rec_color in ['green', 'red', 'blue'] else 'black'}; 
                    text-align: center; 
                    margin-bottom: 20px;
                    opacity: 0.9;">
                    <h2 style="margin:0; color: inherit;">🎯 {rec_status}</h2>
                    <p style="margin:5px 0 0 0; font-size: 16px;">{rec_reason}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"#### 📊 ข้อมูลเจาะลึก: {row['symbol']}")
            
            # Layout: 2 Main Columns (Left: Overview, Right: Analysis & Chart)
            main_c1, main_c2 = st.columns([1, 2.5])
            
            with main_c1:
                st.markdown("**Overview**")
                st.metric(label="ราคาปัจจุบัน", value=f"{row['price']:.2f}")
                st.metric(label="Fair Value (Graham)", value=f"{row['fair_value']:.2f}", delta=f"MOS {row['mos_pct']:.0f}%")
                
                # --- DDM Valuation ---
                ddm_val_calc = None  # Store for MOS section
                try:
                    # Constants
                    rf_rate = 0.025
                    mkt_prem = 0.08
                    g_rate = 0.03 # Conservative Growth 3%
                    
                    # Vars
                    beta_val = row.get('beta', 1.0)
                    if pd.isnull(beta_val): beta_val = 1.0
                    
                    div_yield = row.get('yield', 0)
                    if pd.isnull(div_yield): div_yield = 0
                    
                    # Calc Ke (CAPM) with Safety Floor
                    raw_ke = rf_rate + (beta_val * (mkt_prem - rf_rate)) 
                    ke_val = max(raw_ke, 0.06) # Floor Ke at 6% (Min expected return)

                    d0 = row['price'] * div_yield
                    d1 = d0 * (1 + g_rate)
                    
                    # Ensure Denominator is healthy (Min Spread 2%)
                    spread = ke_val - g_rate
                    
                    if spread >= 0.02 and d0 > 0:
                        ddm_val = d1 / spread
                        ddm_val_calc = ddm_val # STORE IT
                        mos_ddm = ((ddm_val - row['price']) / ddm_val) * 100
                        st.metric(label="Fair Value (DDM)", value=f"{ddm_val:.2f}", delta=f"MOS {mos_ddm:.0f}%", help="DDM (g=3%, Min Ke=6%)")
                    else:
                        st.metric(label="Fair Value (DDM)", value="N/A", help="Ke too low or No Dividend")
                except:
                    pass
                
                st.write("---")
                st.markdown(f"**Grade:** {row['rating_icon']}")
                st.markdown(f"**Type:** {row['quadrant']}")
                st.markdown(f"**Industry:** {row['sector_th']}")
                
                qa_metric = "ROIC" if use_roic else "ROE"
                qa_val = row['roic'] if use_roic else row['roe']
                if pd.notnull(qa_val):
                    st.metric(label=f"Quality ({qa_metric})", value=f"{qa_val:.2f}%")
                else:
                    st.metric(label=f"Quality ({qa_metric})", value="N/A")            
            
            with main_c2:
                # Row 1: F-Score Breakdown (Split into 2 sub-cols)
                sub_c1, sub_c2 = st.columns(2)
                
                pass_icon = "✅"
                fail_icon = "❌"
                
                with sub_c1:
                    st.markdown("**🩺 สุขภาพการเงิน**")
                    # Check 1: ROA
                    roa_val = row.get('return_on_assets', 0) if pd.notnull(row.get('return_on_assets')) else 0
                    check_roa = roa_val > 0
                    st.write(f"{pass_icon if check_roa else fail_icon} **Positive ROA:** {roa_val*100:.2f}%")
                    
                    # Check 2: OCF
                    ocf_val = row.get('ocf', 0) if pd.notnull(row.get('ocf')) else 0
                    check_ocf = ocf_val > 0
                    st.write(f"{pass_icon if check_ocf else fail_icon} **Positive Cash Flow:** {ocf_val/1_000_000:,.1f} M")
                    
                    # Check 3: Accruals (OCF/Assets > ROA)
                    assets = row.get('total_assets', 1)
                    if pd.isnull(assets) or assets == 0: assets = 1
                    ocf_roa = ocf_val / assets
                    check_accruals = ocf_roa > roa_val
                    st.write(f"{pass_icon if check_accruals else fail_icon} **Earn. Qual. (CFO > ROA):** {'Pass' if check_accruals else 'Fail'}")

                    # Check 4: Leverage (D/E < 1.0) # Using Ratio now
                    de_val = row.get('de', 0) if pd.notnull(row.get('de')) else 0
                    de_ratio_disp = de_val # It is already stored as ratio
                    check_de = de_ratio_disp < 1.0
                    st.write(f"{pass_icon if check_de else fail_icon} **Low Debt (D/E < 1):** {de_ratio_disp:.2f}x")
                    
                    # Check 5: Liquidity (Current Ratio > 1.0)
                    cr_val = row.get('current_ratio', 0) if pd.notnull(row.get('current_ratio')) else 0
                    check_cr = cr_val > 1.0
                    st.write(f"{pass_icon if check_cr else fail_icon} **Liquidity (CR > 1):** {cr_val:.2f}x")
                    
                with sub_c2:
                    st.markdown("**📈 ประสิทธิภาพ & มูลค่า**")
                    
                    # Check 6: Efficiency (Rev Growth > 0)
                    rg_val = row.get('revenue_growth', 0) if pd.notnull(row.get('revenue_growth')) else 0
                    check_rg = rg_val > 0
                    st.write(f"{pass_icon if check_rg else fail_icon} **Rev. Growth:** {rg_val*100:.2f}%")
                    
                    # Check 7: Margins (Gross Margin > 0)
                    gm_val = row.get('gross_margins', 0) if pd.notnull(row.get('gross_margins')) else 0
                    check_gm = gm_val > 0
                    st.write(f"{pass_icon if check_gm else fail_icon} **Pos. Gross Margin:** {gm_val*100:.2f}%")

                    # Check 8: Dividend (Yield > 0)
                    yld_val = row.get('yield', 0) if pd.notnull(row.get('yield')) else 0
                    check_yld = yld_val > 0
                    st.write(f"{pass_icon if check_yld else fail_icon} **Pays Dividend:** {yld_val*100:.2f}%")
                    
                    # Check 9: Valuation (PE < 15)
                    pe_val = row.get('pe', 99) 
                    if pd.isnull(pe_val): pe_val = 99
                    check_pe = pe_val < 15
                    st.write(f"{pass_icon if check_pe else fail_icon} **Cheap (PE < 15):** {pe_val:.2f}x")
                    
                    st.info(f"**Total F-Score:** {int(row['f_score'])} / 9")
                
                # Use a container for the chart to ensure it stays within the right column block
                st.markdown("##### 📉 Price Trend (10 Years)")
                try:
                    # Fetch History on demand
                    t = yf.Ticker(row['symbol'])
                    # Fetch 10 Years of history
                    hist = t.history(period="10y")
                    if not hist.empty:
                        # Simple Line Chart
                        st.line_chart(hist['Close'], height=250, width=0, use_container_width=True, color="#2980b9")
                    else:
                        st.warning("No price history available.")
                except:
                    st.error("Could not load price chart.")



            # --- Margin of Safety Scenarios ---
            st.markdown("---")
            st.markdown("#### 🛡️ Margin of Safety Buying Zones")
            
            # Selector
            val_method = st.radio("เลือกโมเดล (Valuation Base):", ["Graham (Assets/EPS)", "DDM (Dividend/Cashflow)"], horizontal=True)
            
            base_fv = 0
            is_valid_method = True
            
            if val_method.startswith("Graham"):
                base_fv = row['fair_value']
            else: # DDM selected
                if ddm_val_calc is not None:
                     base_fv = ddm_val_calc
                else:
                     is_valid_method = False
            
            if is_valid_method and base_fv > 0:
                st.caption(f"ราคาที่ควรซื้อเมื่อเผื่อส่วนลดความปลอดภัย (อ้างอิง Fair Value: {base_fv:.2f})")
                
                # Grids
                m1, m2 = st.columns(2)
                m3, m4 = st.columns(2)
                
                # MOS Prices
                mos30 = base_fv * (1 - 0.30)
                mos40 = base_fv * (1 - 0.40)
                mos50 = base_fv * (1 - 0.50) # Buffett Zone
                mos60 = base_fv * (1 - 0.60) # Deep Value
                
                curr_price = row['price']
                
                # Check Zones
                # Passed = Price is LOWER than MOS level
                p30 = "✅" if curr_price < mos30 else ""
                p40 = "✅" if curr_price < mos40 else ""
                p50 = "✅" if curr_price < mos50 else ""
                p60 = "✅" if curr_price < mos60 else ""
                
                # Highlight "Current Best Zone" (The deepest valid level)
                h30 = h40 = h50 = h60 = ""
                if curr_price < mos60: h60 = "✨ YOU ARE HERE"
                elif curr_price < mos50: h50 = "✨ YOU ARE HERE"
                elif curr_price < mos40: h40 = "👈 YOU ARE HERE"
                elif curr_price < mos30: h30 = "👈 YOU ARE HERE"
                
                with m1:
                    st.error(f"**MOS 30%:** {mos30:.2f} {p30} {h30}")
                with m2:
                    st.warning(f"**MOS 40%:** {mos40:.2f} {p40} {h40}")
                with m3:
                    st.success(f"**MOS 50%:** {mos50:.2f} {p50} {h50}")
                with m4:
                    st.info(f"**MOS 60%:** {mos60:.2f} {p60} {h60}")

                # Current Status Box
                # Recalculate discount based on NEW base_fv
                curr_discount = ((base_fv - row['price']) / base_fv) * 100
                
                st.markdown("")
                if curr_discount > 30:
                    st.success(f"✅ **ราคาดีมาก:** ปัจจุบันลดราคา {curr_discount:.1f}% จากมูลค่าจริง")
                elif curr_discount > 0:
                    st.info(f"🆗 **พอใช้ได้:** ปัจจุบันลดราคา {curr_discount:.1f}% (Margin บาง)")
                else:
                    extra_prem = abs(curr_discount)
                    st.error(f"❌ **แพงไป:** ราคาเกินมูลค่าอยู่ {extra_prem:.1f}%")
            else:
                 st.warning("⚠️ ไม่สามารถคำนวณโมเดลนี้ได้ (ไม่มีปันผล หรือ ข้อมูลไม่เพียงพอ)")

            # --- ROIC vs WACC Section ---
            st.markdown("---")
            st.markdown("#### ⚖️ การสร้างมูลค่า (ROIC vs WACC)")
            
            # WACC Calculation Logic (Conservative)
            # Assumptions
            RF = 0.025 # Risk Free Rate (Thai Bond 10Y ~2.5%)
            RM = 0.08  # Market Premium (~8%)
            KD = 0.045 # Cost of Debt (Pre-tax) ~4.5%
            TAX = 0.20 # Corporate Tax Rate
            
            raw_beta = row.get('beta', 1.0)
            if pd.isnull(raw_beta): raw_beta = 1.0
            
            # 1. Adjust Beta (Clamp outliers)
            # Beta < 0.6 is unrealistic for most stocks (unless defensive/utility). 
            # Beta > 2.0 is extreme volatility.
            adj_beta = max(0.6, min(raw_beta, 2.5))
            
            # 2. Cost of Equity (CAPM) with Floor
            # Ke = Rf + Beta * Rm_Premium
            calc_ke = RF + adj_beta * (RM) 
            ke = max(calc_ke, 0.06) # Floor Ke at 6% (Min Investor Expectation)
            
            # Weights
            # Ensure values are float
            mcap = row.get('price', 0) * row.get('shares', 0)
            
            # Note: 'de' (Debt/Equity) in our DB is stored as Percentage (e.g. 150.0 = 150%)
            raw_de_pct = row.get('de', 0) 
            if pd.isnull(raw_de_pct): raw_de_pct = 50.0 # Default 50%
            
            de_ratio = raw_de_pct / 100.0 # Convert to Decimal Ratio (e.g. 1.5)
            
            # WACC Formula: Ke*We + Kd*(1-Tax)*Wd
            we = 1 / (1 + de_ratio)
            wd = de_ratio / (1 + de_ratio)
            
            wacc = (we * ke) + (wd * KD * (1 - TAX))
            
            # Display
            col_w1, col_w2, col_w3 = st.columns(3)
            
            roic_disp = row.get('roic', 0)
            if pd.isnull(roic_disp): roic_disp = 0
            
            # Unit Correction Logic:
            # If ROIC > 4.0 (400%), it's likely already in % from source (e.g. 9.1 instead of 0.091)
            # Threshold 4.0 (400%) is safe because rarely any company has ROIC > 400%
            if roic_disp > 4.0:
                roic_disp = roic_disp / 100.0
            
            spread = roic_disp - wacc
            
            with col_w1:
                st.metric("ROIC (ผลตอบแทนลงทุน)", f"{roic_disp*100:.2f}%")
            with col_w2:
                st.metric(f"WACC (ต้นทุนเงินทุน) [Beta {adj_beta:.2f}]", f"{wacc*100:.2f}%")
            with col_w3:
                st.metric("Spread (กำไรส่วนเพิ่ม)", f"{spread*100:.2f}%", delta_color="normal", delta=f"{'Created Value ✅' if spread>0 else 'Destroyed Value ❌'}")
            
            if spread > 0:
                st.success(f"**Wealth Creator:** บริษัทนี้สร้างผลตอบแทนได้สูงกว่าต้นทุนทางการเงิน {(spread*100):.2f}% (ยิ่งเยอะยิ่งดี)")
            else:
                st.error(f"**Wealth Destroyer:** บริษัทนี้ทำผลตอบแทนได้ต่ำกว่าต้นทุนทางการเงิน! ยิ่งลงทุนยิ่งเสียมูลค่า")
        
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

        ---
        ### 🕵️‍♀️ คู่มือแปลความหมาย (Rating + Category)
        การจับคู่กันของ **Rating (ดาว)** และ **Category (โซนสี)** บอกอะไรเราได้บ้าง?

        **กลุ่ม 1: หุ้นเกรด A+ (น่าสนใจที่สุด)**
        *   ⭐⭐⭐ + 💎 **ของดีราคาถูก**: **"The Perfect Stock"** งบการเงินแข็งแกร่ง (F-Score สูง) + ราคาถูกกว่ามูลค่าจริงมาก (MOS สูง) + อยู่ในโซนของดี (PE ต่ำ ROE สูง) = **ต้องดูเป็นอันดับแรก!**
        *   ⭐⭐ + 💎 **ของดีราคาถูก**: **"เกือบเทพ"** พื้นฐานดีมาก แต่อาจจะยังไม่ถูกจัด (MOS ต่ำกว่า) หรือคะแนนงบการเงินหย่อนลงมานิดหน่อย แต่น่าลงทุนมาก

        **กลุ่ม 2: หุ้นขัดแย้ง (ต้องทำการบ้านเพิ่ม)**
        *   ⭐⭐⭐ หรือ ⭐⭐ + ⚠️ **ระวังกับดัก**: **"Deep Value / Turnaround"** โปรแกรมมองว่าเป็น "กับดัก" เพราะ ROE ต่ำ (กำไรอาจจะดูไม่สวยในยอดรวม) แต่ไส้ใน **งบแกร่งและราคาถูกมาก** อาจเป็นหุ้นวัฏจักรที่กำลังจะฟื้น หรือมีทรัพย์สินเยอะมาก
        *   😐 + 💎 **ของดีราคาถูก**: **"ดีแต่...เฉยๆ"** อยู่ในโซนของดี (PE ต่ำ ROE สูง) แต่งบการเงิน (F-Score) อาจจะปานกลาง หรือราคาตลาดใกล้เคียงมูลค่าจริงแล้ว (ไม่มีแต้มต่อ MOS)
        *   ⚠️ + 💎 **ของดีราคาถูก**: **"กับดักในคราบทองคำ"** ดูภายนอกเหมือนดี (PE ต่ำ ROE สูง) แต่ไส้ใน **มีความเสี่ยง** เช่น หนี้เยอะเกินไป หรือกระแสเงินสดแย่ (F-Score ต่ำ) **ต้องระวัง!**

        **กลุ่ม 3: หุ้นแพง/หุ้นซิ่ง**
        *   ⭐ หรือ 😐 + ⭐ **แพงแต่ดี**: **"Growth Stock"** หุ้นคุณภาพดี (ROE สูง) ที่คนให้ค่าเยอะ (PE สูง) มักเป็นหุ้นเติบโต ถ้างบการเงินยังดี ก็ถือว่าน่าสนใจแต่ไม่มีแต้มต่อด้านราคา (MOS ต่ำ/ติดลบ)

        **กลุ่ม 4: หุ้นควรเลี่ยง**
        *   ⚠️ + ❌ **แพงและแย่**: **"หนีไป!"** แพงก็แพง (PE สูง) คุณภาพก็แย่ (ROE ต่ำ) แถมมีความเสี่ยงทางการเงิน (F-Score ต่ำ/หนี้เยอะ)
        *   ⭐⭐ + ❌ **แพงและแย่**: **"Asset Play?"** (กรณีหายาก) จัดว่าแพงและแย่ในมุมกำไร (PE สูง, ROE ต่ำ) แต่ Rating ดีได้เพราะราคาต่ำกว่ามูลค่าทางบัญชีมาก (Asset เยอะ)
        """)
    else:
        st.error("No valid data points found after filtering. This might happen if Yahoo Finance data is temporarily unavailable or incomplete.")

