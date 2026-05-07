import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Hollywood Movies Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main background - Dark red to black */
    .main {
        background: linear-gradient(135deg, #1a0000 0%, #330000 30%, #4d0000 100%);
    }
    
    /* Sidebar - Deep red */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0000 0%, #1a0000 100%);
        border-right: 3px solid #ff4444;
    }
    
    /* Metrics - Bright red */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #ff4444;
        text-shadow: 0 0 20px rgba(255, 68, 68, 0.6);
    }
    
    [data-testid="stMetricLabel"] {
        color: #cc9999;
        font-weight: 600;
    }
    
    /* Headers - Fire gradient */
    h1 {
        background: linear-gradient(90deg, #ff4444 0%, #ff6b35 50%, #ffa500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        filter: drop-shadow(0 0 25px rgba(255, 68, 68, 0.4));
    }
    
    h2, h3 {
        color: #ff6b35 !important;
        font-weight: 700 !important;
        text-shadow: 0 0 15px rgba(255, 107, 53, 0.3);
    }
    
    /* Tabs - Fiery theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, rgba(26, 0, 0, 0.8) 0%, rgba(51, 0, 0, 0.8) 100%);
        border-radius: 15px;
        padding: 8px;
        border: 2px solid rgba(255, 68, 68, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(255, 68, 68, 0.15) 0%, rgba(255, 107, 53, 0.15) 100%);
        border-radius: 10px;
        color: #ff6b35;
        padding: 12px 24px;
        font-weight: 600;
        border: 1px solid rgba(255, 68, 68, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff4444 0%, #ff6b35 50%, #ffa500 100%);
        color: #ffffff;
        box-shadow: 0 4px 20px rgba(255, 68, 68, 0.5);
    }
    
    /* Buttons - Fire gradient */
    .stButton>button {
        background: linear-gradient(135deg, #ff4444 0%, #ff6b35 50%, #ffa500 100%);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(255, 68, 68, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 30px rgba(255, 68, 68, 0.6);
    }
    
    /* Metric cards - Dark red with fire border */
    .metric-card {
        background: linear-gradient(135deg, #1a0000 0%, #330000 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
        text-align: center;
        border: 2px solid #ff4444;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('hollywood_movies.csv')
        
        # Remove any unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                temp = df[col].astype(str).str.replace('$', '', regex=False)
                temp = temp.str.replace(',', '', regex=False)
                temp = temp.str.replace('%', '', regex=False)
                temp = pd.to_numeric(temp, errors='ignore')
                if pd.api.types.is_numeric_dtype(temp):
                    df[col] = temp
        
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("❌ Could not load data")
    st.stop()

# Identify columns
all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

movie_col = next((c for c in all_cols if 'movie' in c.lower() or 'film' in c.lower() or 'title' in c.lower()), all_cols[0] if all_cols else None)
year_col = next((c for c in all_cols if 'year' in c.lower()), None)
genre_col = next((c for c in all_cols if 'genre' in c.lower()), None)

# Sidebar
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='color: white; font-size: 32px;'>🎬</h1>
    <h2 style='color: white; font-size: 24px; margin: 0;'>Hollywood</h2>
    <h3 style='color: #a0a0c0; font-size: 16px; margin: 0;'>Analytics Dashboard</h3>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Data Explorer", 
        "📈 Visual Analytics",
        "📉 Statistics Lab",
        "🎯 Confidence Intervals",
        "🎲 Probability Engine",
        "🔮 AI Predictions",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Smart Filters")

filtered_df = df.copy()

if genre_col:
    genres = sorted([g for g in df[genre_col].dropna().unique() if str(g) != 'nan'])
    if genres:
        selected_genres = st.sidebar.multiselect("🎭 Genres", genres, default=genres)
        filtered_df = filtered_df[filtered_df[genre_col].isin(selected_genres)]

if year_col:
    year_data = pd.to_numeric(df[year_col], errors='coerce').dropna()
    if len(year_data) > 0:
        min_year = int(year_data.min())
        max_year = int(year_data.max())
        year_range = st.sidebar.slider("📅 Year Range", min_year, max_year, (min_year, max_year))
        filtered_df = filtered_df[
            (pd.to_numeric(filtered_df[year_col], errors='coerce') >= year_range[0]) & 
            (pd.to_numeric(filtered_df[year_col], errors='coerce') <= year_range[1])
        ]

st.sidebar.success(f"✅ {len(filtered_df)} movies")

filtered_numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# DASHBOARD
# ============================================================================

if page == "🏠 Dashboard":
<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 30px;'>
    <h1 style='font-size: 48px; color: white; margin: 0;'>🎬 Hollywood Movies Analytics</h1>
    <p style='color: rgba(255,255,255,0.9); font-size: 20px; margin: 10px 0 0 0;'>Advanced Statistical Analysis & Prediction System</p>
    <p style='color: rgba(255,255,255,0.7); font-size: 14px;'>Probability & Statistics | Spring 2026</p>
</div>

# Dashboard Header
st.markdown("""
<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #1a0000 0%, #4d0000 100%); border-radius: 20px; margin-bottom: 30px; border: 3px solid #ff4444; box-shadow: 0 0 40px rgba(255, 68, 68, 0.4);'>
    <h1 style='font-size: 48px; background: linear-gradient(90deg, #ff4444 0%, #ff6b35 50%, #ffa500 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; filter: drop-shadow(0 0 25px rgba(255, 68, 68, 0.6));'>🎬 Hollywood Movies Analytics</h1>
    <p style='color: #ff6b35; font-size: 20px; margin: 10px 0 0 0; font-weight: 600;'>Advanced Statistical Analysis & Prediction System</p>
    <p style='color: #cc9999; font-size: 14px;'>Probability & Statistics | Spring 2026</p>
</div>
""", unsafe_allow_html=True)
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='margin: 0; font-size: 36px;'>🎬</h2>
            <h3 style='margin: 10px 0; font-size: 32px; font-weight: 800;'>{len(filtered_df):,}</h3>
            <p style='margin: 0; font-size: 14px; opacity: 0.9;'>Total Movies</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if len(filtered_numeric_cols) > 0:
            col_name = filtered_numeric_cols[0]
            avg_val = filtered_df[col_name].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='margin: 0; font-size: 36px;'>⭐</h2>
                <h3 style='margin: 10px 0; font-size: 32px; font-weight: 800;'>{avg_val:.1f}</h3>
                <p style='margin: 0; font-size: 14px; opacity: 0.9;'>Avg {col_name}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if len(filtered_numeric_cols) > 1:
            col_name = filtered_numeric_cols[1]
            avg_val = filtered_df[col_name].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='margin: 0; font-size: 36px;'>📈</h2>
                <h3 style='margin: 10px 0; font-size: 32px; font-weight: 800;'>{avg_val:.1f}</h3>
                <p style='margin: 0; font-size: 14px; opacity: 0.9;'>Avg {col_name}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if genre_col:
            genre_count = filtered_df[genre_col].nunique()
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='margin: 0; font-size: 36px;'>🎭</h2>
                <h3 style='margin: 10px 0; font-size: 32px; font-weight: 800;'>{genre_count}</h3>
                <p style='margin: 0; font-size: 14px; opacity: 0.9;'>Genres</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if len(filtered_numeric_cols) >= 2 and genre_col:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Genre Performance Comparison")
            
            genre_stats = filtered_df.groupby(genre_col)[filtered_numeric_cols[0]].mean().sort_values(ascending=False).head(8)
            
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            colors = plt.cm.viridis(np.linspace(0, 1, len(genre_stats)))
            bars = ax.barh(genre_stats.index, genre_stats.values, color=colors, edgecolor='black', linewidth=1.5)
            
            for i, (idx, val) in enumerate(genre_stats.items()):
                ax.text(val, i, f' {val:.1f}', va='center', fontsize=10, fontweight='bold')
            
            ax.set_xlabel(filtered_numeric_cols[0], fontsize=12, fontweight='bold')
            ax.set_ylabel('Genre', fontsize=12, fontweight='bold')
            ax.set_title(f'Average {filtered_numeric_cols[0]} by Genre', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 📈 Distribution Analysis")
            
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            
            data_for_violin = []
            labels_for_violin = []
            
            top_genres = filtered_df[genre_col].value_counts().head(6).index
            
            for genre in top_genres:
                genre_data = filtered_df[filtered_df[genre_col] == genre][filtered_numeric_cols[1]].dropna()
                if len(genre_data) > 0:
                    data_for_violin.append(genre_data)
                    labels_for_violin.append(genre)
            
            if data_for_violin:
                parts = ax.violinplot(data_for_violin, positions=range(len(data_for_violin)), 
                                     showmeans=True, showmedians=True)
                
                colors = plt.cm.plasma(np.linspace(0, 1, len(data_for_violin)))
                for pc, color in zip(parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax.set_xticks(range(len(labels_for_violin)))
                ax.set_xticklabels(labels_for_violin, rotation=45, ha='right')
                ax.set_ylabel(filtered_numeric_cols[1], fontsize=12, fontweight='bold')
                ax.set_title(f'Distribution of {filtered_numeric_cols[1]} by Genre', fontsize=14, fontweight='bold', pad=20)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎬 Featured Movies")
    
    display_cols = [c for c in [movie_col, year_col, genre_col] + filtered_numeric_cols[:4] if c]
    preview_df = filtered_df[display_cols].head(10).copy()
    
    st.dataframe(preview_df, use_container_width=True, hide_index=True, height=400)
    
    if len(filtered_numeric_cols) >= 2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🏆 Top 5 by {filtered_numeric_cols[0]}")
            top_data = filtered_df.nlargest(5, filtered_numeric_cols[0])
            display = [movie_col, filtered_numeric_cols[0]]
            if genre_col:
                display.insert(1, genre_col)
            
            top_display = top_data[display].copy()
            top_display.index = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
            st.dataframe(top_display, use_container_width=True)
        
        with col2:
            st.markdown(f"### 💎 Top 5 by {filtered_numeric_cols[1]}")
            top_data = filtered_df.nlargest(5, filtered_numeric_cols[1])
            display = [movie_col, filtered_numeric_cols[1]]
            if genre_col:
                display.insert(1, genre_col)
            
            top_display = top_data[display].copy()
            top_display.index = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
            st.dataframe(top_display, use_container_width=True)

# ============================================================================
# DATA EXPLORER
# ============================================================================

elif page == "📊 Data Explorer":
    st.markdown("# 📊 Data Explorer")
    
    tab1, tab2, tab3 = st.tabs(["📋 Dataset Overview", "🔍 Column Analysis", "✅ Data Quality"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Rows", f"{filtered_df.shape[0]:,}")
        with col2:
            st.metric("📝 Total Columns", filtered_df.shape[1])
        with col3:
            st.metric("🔢 Numeric Columns", len(filtered_numeric_cols))
        with col4:
            st.metric("💾 Memory Usage", f"{filtered_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        st.markdown("---")
        st.markdown("### Complete Dataset")
        st.dataframe(filtered_df, use_container_width=True, height=500)
        
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Filtered Data (CSV)",
            csv,
            "hollywood_movies_filtered.csv",
            "text/csv",
            key='download-csv'
        )
    
    with tab2:
        st.markdown("### Detailed Column Information")
        
        col_info = pd.DataFrame({
            'Column': filtered_df.columns,
            'Type': filtered_df.dtypes.astype(str).values,
            'Non-Null': filtered_df.count().values,
            'Null': filtered_df.isnull().sum().values,
            'Unique': [filtered_df[col].nunique() for col in filtered_df.columns]
        })
        
        st.dataframe(col_info, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("### Data Quality Report")
        
        missing = pd.DataFrame({
            'Column': filtered_df.columns,
            'Missing': filtered_df.isnull().sum().values,
            'Missing %': (filtered_df.isnull().sum().values / len(filtered_df) * 100).round(2)
        })
        
        missing = missing[missing['Missing'] > 0].sort_values('Missing %', ascending=False)
        
        if len(missing) == 0:
            st.success("✅ Perfect! No missing values detected in the dataset.")
        else:
            st.warning(f"⚠️ Found missing values in {len(missing)} column(s)")
            st.dataframe(missing, use_container_width=True, hide_index=True)

# ============================================================================
# VISUAL ANALYTICS (Keeping previous implementation)
# ============================================================================

elif page == "📈 Visual Analytics":
    st.markdown("# 📈 Advanced Visual Analytics")
    
    if len(filtered_numeric_cols) == 0:
        st.warning("⚠️ No numeric columns available")
        st.stop()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📦 Comparisons", "🔵 Relationships", "🔥 Correlations"])
    
    with tab1:
        st.markdown("### Distribution Analysis")
        
        col = st.selectbox("Select Variable", filtered_numeric_cols, key='dist')
        data = filtered_df[col].dropna()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), facecolor='white')
            
            ax1.hist(data, bins=30, color='#667eea', alpha=0.7, edgecolor='black', density=True)
            
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            ax1.set_xlabel(col, fontsize=12, fontweight='bold')
            ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax1.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold', pad=20)
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            bp = ax2.boxplot([data], vert=False, patch_artist=True, widths=0.5)
            bp['boxes'][0].set_facecolor('#764ba2')
            bp['boxes'][0].set_alpha(0.7)
            
            ax2.set_xlabel(col, fontsize=12, fontweight='bold')
            ax2.set_title('Box Plot', fontsize=14, fontweight='bold', pad=20)
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 📊 Stats")
            st.metric("Mean", f"{data.mean():.2f}")
            st.metric("Median", f"{data.median():.2f}")
            st.metric("Std Dev", f"{data.std():.2f}")
            st.metric("Min", f"{data.min():.2f}")
            st.metric("Max", f"{data.max():.2f}")
            st.metric("Count", f"{len(data):,}")
    
    with tab2:
        st.markdown("### Genre Comparison")
        
        if not genre_col:
            st.info("No genre column found")
        else:
            col = st.selectbox("Select Metric", filtered_numeric_cols, key='comp')
            
            genre_stats = filtered_df.groupby(genre_col)[col].agg(['mean', 'std']).reset_index()
            genre_stats = genre_stats.sort_values('mean', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
            
            x_pos = np.arange(len(genre_stats))
            colors = plt.cm.viridis(np.linspace(0, 1, len(genre_stats)))
            
            bars = ax.bar(x_pos, genre_stats['mean'], yerr=genre_stats['std'], 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(genre_stats[genre_col], rotation=45, ha='right')
            ax.set_ylabel(f'Average {col}', fontsize=12, fontweight='bold')
            ax.set_title(f'Mean {col} by Genre', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, genre_stats['mean']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab3:
        st.markdown("### Relationship Analysis")
        
        if len(filtered_numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("X Variable", filtered_numeric_cols, key='x_rel')
            with col2:
                y_var = st.selectbox("Y Variable", filtered_numeric_cols, 
                                   index=min(1, len(filtered_numeric_cols)-1), key='y_rel')
            
            scatter_data = filtered_df[[x_var, y_var]].dropna()
            
            if len(scatter_data) > 0:
                correlation = scatter_data[x_var].corr(scatter_data[y_var])
                
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
                
                ax.scatter(scatter_data[x_var], scatter_data[y_var],
                          alpha=0.6, s=80, c=scatter_data[y_var],
                          cmap='viridis', edgecolors='black', linewidth=0.5)
                
                z = np.polyfit(scatter_data[x_var], scatter_data[y_var], 1)
                p = np.poly1d(z)
                ax.plot(scatter_data[x_var].sort_values(), 
                       p(scatter_data[x_var].sort_values()),
                       "r--", linewidth=2, label=f'r={correlation:.3f}')
                
                ax.set_xlabel(x_var, fontsize=12, fontweight='bold')
                ax.set_ylabel(y_var, fontsize=12, fontweight='bold')
                ax.set_title(f'{x_var} vs {y_var}', fontsize=14, fontweight='bold', pad=20)
                ax.legend()
                ax.grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                if abs(correlation) > 0.7:
                    st.success(f"🔥 **Strong correlation** detected (r = {correlation:.3f})")
                elif abs(correlation) > 0.4:
                    st.info(f"📊 **Moderate correlation** detected (r = {correlation:.3f})")
                else:
                    st.warning(f"📉 **Weak correlation** detected (r = {correlation:.3f})")
    
    with tab4:
        st.markdown("### Correlation Matrix")
        
        if len(filtered_numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns")
        else:
            corr_data = filtered_df[filtered_numeric_cols].dropna()
            corr = corr_data.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
            
            im = ax.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            ax.set_xticks(np.arange(len(corr.columns)))
            ax.set_yticks(np.arange(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr.columns)
            
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    value = corr.iloc[i, j]
                    color = 'white' if abs(value) > 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha="center", va="center", 
                           color=color, fontsize=9, fontweight='bold')
            
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)

# ============================================================================
# STATISTICS LAB
# ============================================================================

elif page == "📉 Statistics Lab":
    st.markdown("# 📉 Advanced Statistics Laboratory")
    
    if len(filtered_numeric_cols) == 0:
        st.warning("No numeric columns available")
        st.stop()
    
    tab1, tab2 = st.tabs(["📊 Summary Statistics", "🔬 Detailed Analysis"])
    
    with tab1:
        st.markdown("### Summary Statistics")
        
        summary = filtered_df[filtered_numeric_cols].describe()
        st.dataframe(summary.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
        
        csv = summary.to_csv()
        st.download_button("📥 Download Statistics", csv, "statistics.csv", "text/csv")
    
    with tab2:
        st.markdown("### Detailed Analysis")
        
        col = st.selectbox("Select Variable", filtered_numeric_cols)
        data = filtered_df[col].dropna()
        
        mean = data.mean()
        median = data.median()
        mode_val = data.mode().iloc[0] if len(data.mode()) > 0 else np.nan
        variance = data.var()
        std_dev = data.std()
        minimum = data.min()
        maximum = data.max()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### Central Tendency")
            st.metric("Mean", f"{mean:.4f}")
            st.metric("Median", f"{median:.4f}")
            st.metric("Mode", f"{mode_val:.4f}")
        
        with col2:
            st.markdown("#### Dispersion")
            st.metric("Variance", f"{variance:.4f}")
            st.metric("Std Dev", f"{std_dev:.4f}")
            st.metric("Range", f"{maximum-minimum:.4f}")
        
        with col3:
            st.markdown("#### Position")
            st.metric("Min", f"{minimum:.4f}")
            st.metric("Q1", f"{q1:.4f}")
            st.metric("Q3", f"{q3:.4f}")
        
        with col4:
            st.markdown("#### Shape")
            st.metric("Max", f"{maximum:.4f}")
            st.metric("IQR", f"{iqr:.4f}")
            st.metric("Skewness", f"{skewness:.4f}")
        
        st.metric("Kurtosis", f"{kurtosis:.4f}")

# ============================================================================
# CONFIDENCE INTERVALS
# ============================================================================

elif page == "🎯 Confidence Intervals":
    st.markdown("# 🎯 Confidence Interval Calculator")
    
    if len(filtered_numeric_cols) == 0:
        st.warning("No numeric columns available")
        st.stop()
    
    col = st.selectbox("Select Variable", filtered_numeric_cols)
    confidence = st.slider("Confidence Level (%)", 90, 99, 95, 1)
    
    data = filtered_df[col].dropna()
    
    if len(data) < 2:
        st.error("Not enough data points")
        st.stop()
    
    mean = data.mean()
    sem = stats.sem(data)
    ci = stats.t.interval(confidence/100, len(data)-1, loc=mean, scale=sem)
    margin = ci[1] - mean
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Sample Mean", f"{mean:.4f}")
    with col2:
        st.metric("📉 Std Error", f"{sem:.4f}")
    with col3:
        st.metric("⬇️ Lower Limit", f"{ci[0]:.4f}")
    with col4:
        st.metric("⬆️ Upper Limit", f"{ci[1]:.4f}")
    
    st.success(f"""
    ### ✅ Confidence Interval Result
    
    We are **{confidence}% confident** that the true population mean (μ) of **{col}** 
    lies between **{ci[0]:.4f}** and **{ci[1]:.4f}**.
    
    **Margin of Error:** ±{margin:.4f}  
    **Sample Size:** {len(data)} observations
    """)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
    
    ax.axvline(mean, color='red', linewidth=3, label=f'Sample Mean: {mean:.2f}', zorder=3)
    ax.axvline(ci[0], color='green', linewidth=2, linestyle='--', label=f'Lower CI: {ci[0]:.2f}')
    ax.axvline(ci[1], color='blue', linewidth=2, linestyle='--', label=f'Upper CI: {ci[1]:.2f}')
    ax.axvspan(ci[0], ci[1], alpha=0.3, color='yellow', label=f'{confidence}% CI Region')
    
    ax.set_xlabel(col, fontsize=14, fontweight='bold')
    ax.set_title(f'{confidence}% Confidence Interval for Population Mean', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_yticks([])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("""
    **What does this mean?**
    
    If we were to take many samples and calculate confidence intervals for each, 
    approximately {0}% of those intervals would contain the true population mean.
    """.format(confidence))

# ============================================================================
# PROBABILITY ENGINE
# ============================================================================

elif page == "🎲 Probability Engine":
    st.markdown("# 🎲 Probability Distribution Engine")
    
    if len(filtered_numeric_cols) == 0:
        st.warning("No numeric columns available")
        st.stop()
    
    tab1, tab2 = st.tabs(["📊 Normal Distribution", "🎯 Binomial Probability"])
    
    with tab1:
        st.markdown("### Normal Distribution Analysis")
        
        col = st.selectbox("Select Variable", filtered_numeric_cols, key='norm_var')
        data = filtered_df[col].dropna()
        
        if len(data) < 2:
            st.error("Not enough data")
            st.stop()
        
        mean = data.mean()
        std = data.std()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Mean (μ)", f"{mean:.4f}")
        with col2:
            st.metric("📏 Std Dev (σ)", f"{std:.4f}")
        with col3:
            st.metric("📈 Sample Size", f"{len(data):,}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        
        # Histogram
        n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.7, 
                                   color='skyblue', edgecolor='black', label='Data')
        
        # Normal curve
        x = np.linspace(data.min(), data.max(), 200)
        y = stats.norm.pdf(x, mean, std)
        ax.plot(x, y, 'r-', linewidth=3, label='Normal Distribution')
        
        ax.set_xlabel(col, fontsize=13, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
        ax.set_title(f'Normal Distribution Fit: {col}', fontsize=15, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Probability calculator
        st.markdown("---")
        st.markdown("### 🎲 Probability Calculator")
        
        st.write("Calculate the probability that a value falls within a specified range:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lower = st.number_input(
                "Lower Bound (a)",
                value=float(data.min()),
                format="%.2f",
                help="Enter the lower value of the range"
            )
        
        with col2:
            upper = st.number_input(
                "Upper Bound (b)",
                value=float(data.mean()),
                format="%.2f",
                help="Enter the upper value of the range"
            )
        
        if lower >= upper:
            st.error("❌ Lower bound must be less than upper bound!")
        else:
            # Calculate z-scores
            z_lower = (lower - mean) / std
            z_upper = (upper - mean) / std
            
            # Calculate probability
            prob = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
            
            # Display result
            st.success(f"""
            ### 🎯 Probability Result
            
            **P({lower:.2f} < X < {upper:.2f}) = {prob:.4f}**
            
            This means there is a **{prob*100:.2f}%** probability that a randomly 
            selected movie will have a **{col}** value between **{lower:.2f}** and **{upper:.2f}**.
            
            **Z-scores:**
            - Lower: z = {z_lower:.3f}
            - Upper: z = {z_upper:.3f}
            """)
            
            # Visualize probability
            fig2, ax2 = plt.subplots(figsize=(12, 6), facecolor='white')
            
            x_range = np.linspace(mean - 4*std, mean + 4*std, 500)
            y_range = stats.norm.pdf(x_range, mean, std)
            
            ax2.plot(x_range, y_range, 'b-', linewidth=2, label='Normal Distribution')
            
            # Shade the probability region
            x_fill = x_range[(x_range >= lower) & (x_range <= upper)]
            y_fill = stats.norm.pdf(x_fill, mean, std)
            ax2.fill_between(x_fill, y_fill, alpha=0.5, color='green', 
                            label=f'P({lower:.1f} < X < {upper:.1f}) = {prob:.4f}')
            
            ax2.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean:.2f}')
            ax2.axvline(lower, color='orange', linestyle='--', linewidth=1.5, label=f'Lower = {lower:.2f}')
            ax2.axvline(upper, color='purple', linestyle='--', linewidth=1.5, label=f'Upper = {upper:.2f}')
            
            ax2.set_xlabel(col, fontsize=12, fontweight='bold')
            ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
            ax2.set_title('Probability Region Visualization', fontsize=14, fontweight='bold', pad=20)
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Normality test
        st.markdown("---")
        st.markdown("### 🧪 Normality Test (Shapiro-Wilk)")
        
        sample_size = min(len(data), 5000)
        sample_data = data.sample(sample_size, random_state=42)
        
        stat, p_value = stats.shapiro(sample_data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Test Statistic", f"{stat:.6f}")
        with col2:
            st.metric("P-value", f"{p_value:.6f}")
        with col3:
            st.metric("Sample Size", f"{sample_size:,}")
        
        if p_value > 0.05:
            st.success("✅ **Data appears to be normally distributed** (p > 0.05)")
            st.write("We fail to reject the null hypothesis. The data follows a normal distribution.")
        else:
            st.warning("⚠️ **Data may not be normally distributed** (p ≤ 0.05)")
            st.write("We reject the null hypothesis. The data does not follow a perfect normal distribution.")
        
        st.info("""
        **Interpretation:**
        - **p > 0.05:** Data is normally distributed
        - **p ≤ 0.05:** Data is NOT normally distributed
        - The test uses a sample of up to 5000 points for computational efficiency
        """)
    
    with tab2:
        st.markdown("### Binomial Probability Analysis")
        
        if len(filtered_numeric_cols) == 0:
            st.warning("No numeric columns available")
        else:
            st.write("""
            Binomial distribution models the number of successes in a fixed number of independent trials.
            Let's analyze the probability of movies being "highly rated" or meeting a certain threshold.
            """)
            
            col = st.selectbox("Select Variable for Analysis", filtered_numeric_cols, key='binom_var')
            
            data = filtered_df[col].dropna()
            
            if len(data) < 2:
                st.error("Not enough data")
                st.stop()
            
            # Define threshold
            st.markdown("### 🎯 Define Success Criterion")
            
            col1, col2 = st.columns(2)
            
            with col1:
                default_threshold = float(data.median())
                threshold = st.slider(
                    f"Success Threshold for {col}",
                    float(data.min()),
                    float(data.max()),
                    default_threshold,
                    help="Values above this threshold are considered 'successes'"
                )
            
            with col2:
                st.metric("📊 Selected Threshold", f"{threshold:.2f}")
            
            # Calculate success probability
            p = (data > threshold).mean()
            
            st.info(f"""
            ### 📈 Success Probability
            
            **p = {p:.4f}** ({p*100:.2f}%)
            
            This means {p*100:.1f}% of movies have a **{col}** value greater than **{threshold:.2f}**.
            
            - **Successes:** {(data > threshold).sum()} movies
            - **Failures:** {(data <= threshold).sum()} movies
            - **Total:** {len(data)} movies
            """)
            
            # Binomial parameters
            st.markdown("---")
            st.markdown("### 🎲 Binomial Probability Calculator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n = st.number_input(
                    "Number of trials (n)",
                    min_value=1,
                    max_value=200,
                    value=10,
                    help="How many movies are you selecting?"
                )
            
            with col2:
                k = st.number_input(
                    "Number of successes (k)",
                    min_value=0,
                    max_value=int(n),
                    value=min(5, int(n)),
                    help="How many successes do you want to calculate probability for?"
                )
            
            # Calculate probabilities
            prob_exact = stats.binom.pmf(k, n, p)
            prob_at_most = stats.binom.cdf(k, n, p)
            prob_at_least = 1 - stats.binom.cdf(k - 1, n, p)
            prob_less_than = stats.binom.cdf(k - 1, n, p)
            prob_more_than = 1 - stats.binom.cdf(k, n, p)
            
            # Display results
            st.markdown("### 📊 Probability Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"P(X = {k})",
                    f"{prob_exact:.4f}",
                    help="Probability of exactly k successes"
                )
                st.caption(f"Exactly {k} successes: **{prob_exact*100:.2f}%**")
            
            with col2:
                st.metric(
                    f"P(X ≤ {k})",
                    f"{prob_at_most:.4f}",
                    help="Probability of at most k successes"
                )
                st.caption(f"At most {k} successes: **{prob_at_most*100:.2f}%**")
            
            with col3:
                st.metric(
                    f"P(X ≥ {k})",
                    f"{prob_at_least:.4f}",
                    help="Probability of at least k successes"
                )
                st.caption(f"At least {k} successes: **{prob_at_least*100:.2f}%**")
            
            # Additional probabilities
            col4, col5 = st.columns(2)
            
            with col4:
                st.metric(
                    f"P(X < {k})",
                    f"{prob_less_than:.4f}",
                    help="Probability of less than k successes"
                )
                st.caption(f"Less than {k} successes: **{prob_less_than*100:.2f}%**")
            
            with col5:
                st.metric(
                    f"P(X > {k})",
                    f"{prob_more_than:.4f}",
                    help="Probability of more than k successes"
                )
                st.caption(f"More than {k} successes: **{prob_more_than*100:.2f}%**")
            
            # Visualization
            st.markdown("---")
            st.markdown("### 📊 Binomial Distribution Visualization")
            
            fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
            
            x_vals = np.arange(0, n + 1)
            probs = stats.binom.pmf(x_vals, n, p)
            
            # Color bars
            colors = ['red' if x == k else 'lightblue' for x in x_vals]
            bars = ax.bar(x_vals, probs, color=colors, edgecolor='black', alpha=0.7, linewidth=1.5)
            
            # Highlight the selected value
            ax.axvline(k, color='darkred', linewidth=2, linestyle='--', 
                      label=f'k = {k} (P = {prob_exact:.4f})')
            
            # Add mean line
            expected_value = n * p
            ax.axvline(expected_value, color='green', linewidth=2, linestyle='--', 
                      label=f'Expected Value = {expected_value:.2f}')
            
            # Labels on bars
            for i, (x, prob) in enumerate(zip(x_vals, probs)):
                if prob > 0.01:  # Only label significant probabilities
                    ax.text(x, prob, f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Number of Successes (k)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Probability P(X = k)', fontsize=13, fontweight='bold')
            ax.set_title(f'Binomial Distribution (n={n}, p={p:.4f})', 
                        fontsize=15, fontweight='bold', pad=20)
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3, axis='y')
            ax.set_xticks(x_vals)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistical summary
            st.markdown("---")
            st.markdown("### 📈 Distribution Statistics")
            
            mean_binom = n * p
            variance_binom = n * p * (1 - p)
            std_binom = np.sqrt(variance_binom)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expected Value (μ)", f"{mean_binom:.2f}")
            with col2:
                st.metric("Variance (σ²)", f"{variance_binom:.2f}")
            with col3:
                st.metric("Std Deviation (σ)", f"{std_binom:.2f}")
            with col4:
                st.metric("Mode", f"{int(np.floor((n+1)*p))}")
            
            st.info(f"""
            **Interpretation:**
            
            In {n} randomly selected movies:
            - We **expect** approximately **{mean_binom:.1f}** movies to have {col} > {threshold:.2f}
            - The **standard deviation** is **{std_binom:.2f}**
            - The most likely number of successes is **{int(np.floor((n+1)*p))}**
            
            **Real-world meaning:**
            If you randomly select {n} movies from this dataset, there's a:
            - **{prob_exact*100:.1f}%** chance of getting exactly {k} movies with {col} > {threshold:.2f}
            - **{prob_at_least*100:.1f}%** chance of getting {k} or more such movies
            - **{prob_at_most*100:.1f}%** chance of getting {k} or fewer such movies
            """)

# ============================================================================
# AI PREDICTIONS
# ============================================================================

elif page == "🔮 AI Predictions":
    st.markdown("# 🔮 AI-Powered Prediction System")
    st.markdown("### Build Machine Learning Models to Predict Movie Performance")
    
    if len(filtered_numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for prediction")
        st.stop()
    
    st.info("""
    **How it works:**
    1. Select a variable you want to predict (target)
    2. Choose features (independent variables) to base predictions on
    3. The AI builds a linear regression model
    4. Evaluate model performance
    5. Make predictions for new movies
    """)
    
    # Model configuration
    st.markdown("---")
    st.markdown("### ⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target = st.selectbox(
            "🎯 Select Target Variable (Y) to Predict",
            filtered_numeric_cols,
            help="This is what you want to predict"
        )
    
    with col2:
        test_size = st.slider(
            "📊 Test Set Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Percentage of data reserved for testing"
        )
    
    available_features = [c for c in filtered_numeric_cols if c != target]
    
    if len(available_features) == 0:
        st.error("No other numeric columns available as features")
        st.stop()
    
    # Default feature selection
    default_features = available_features[:min(3, len(available_features))]
    
    features = st.multiselect(
        "📋 Select Independent Variables (X) - Features",
        available_features,
        default=default_features,
        help="These variables will be used to predict the target"
    )
    
    if len(features) == 0:
        st.warning("⚠️ Please select at least one feature variable")
        st.stop()
    
    # Prepare data
    model_data = filtered_df[[target] + features].dropna()
    
    if len(model_data) < 10:
        st.error(f"❌ Not enough data after removing missing values. Only {len(model_data)} complete records found.")
        st.info("Try selecting different variables or adjusting your filters")
        st.stop()
    
    st.success(f"✅ Using {len(model_data)} movies with complete data for all selected variables")
    
    # Train model
    X = model_data[features]
    y = model_data[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size/100, 
        random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    # Display performance
    st.markdown("---")
    st.markdown("### 📊 Model Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🎯 R² Score (Test)",
            f"{r2_test:.4f}",
            help="Proportion of variance explained (0-1, higher is better)"
        )
        if r2_test > 0.7:
            st.success("Excellent fit!")
        elif r2_test > 0.5:
            st.info("Good fit")
        else:
            st.warning("Moderate fit")
    
    with col2:
        st.metric(
            "📈 R² Score (Train)",
            f"{r2_train:.4f}",
            help="Model fit on training data"
        )
    
    with col3:
        st.metric(
            "📉 RMSE (Test)",
            f"{rmse_test:.2f}",
            help="Root Mean Squared Error (lower is better)"
        )
    
    with col4:
        st.metric(
            "📏 MAE (Test)",
            f"{mae_test:.2f}",
            help="Mean Absolute Error (lower is better)"
        )
    
    with col5:
        overfitting = abs(r2_train - r2_test)
        st.metric(
            "⚠️ Overfit Check",
            f"{overfitting:.4f}",
            help="Difference between train and test R². Lower is better."
        )
        if overfitting < 0.1:
            st.success("No overfitting")
        else:
            st.warning("Possible overfitting")
    
    # Performance interpretation
    st.info(f"""
    **Model Performance Summary:**
    
    - The model explains **{r2_test*100:.1f}%** of the variance in {target}
    - Average prediction error: **±{mae_test:.2f}** units
    - Training size: **{len(X_train)}** movies | Test size: **{len(X_test)}** movies
    """)
    
    # Regression equation
    st.markdown("---")
    st.markdown("### 📐 Regression Equation")
    
    equation = f"**{target}** = {model.intercept_:.4f}"
    for feat, coef in zip(features, model.coef_):
        sign = "+" if coef >= 0 else ""
        equation += f" {sign} {coef:.4f} × **{feat}**"
    
    st.code(equation, language="text")
    
    # Feature importance
    st.markdown("### 🔑 Feature Importance Analysis")
    
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_,
        'Abs Coefficient': np.abs(model.coef_)
    }).sort_values('Abs Coefficient', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
    bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
    
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax.set_title('Feature Coefficients (Green=Positive, Red=Negative)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, coef_df['Coefficient']):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f' {val:.4f}',
               ha='left' if width > 0 else 'right', va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.dataframe(coef_df[['Feature', 'Coefficient']], use_container_width=True, hide_index=True)
    
    st.info("""
    **How to interpret coefficients:**
    - **Positive coefficient:** When this feature increases, the target increases
    - **Negative coefficient:** When this feature increases, the target decreases
    - **Larger absolute value:** Stronger influence on the prediction
    """)
    
    # Actual vs Predicted plot
    st.markdown("---")
    st.markdown("### 📈 Model Accuracy Visualization")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
    
    # Scatter plot
    ax1.scatter(y_test, y_pred_test, alpha=0.6, s=80, c=y_pred_test, 
               cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
            label='Perfect Prediction')
    
    ax1.set_xlabel(f'Actual {target}', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'Predicted {target}', fontsize=12, fontweight='bold')
    ax1.set_title(f'Actual vs Predicted {target}', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Residual plot
    residuals = y_test - y_pred_test
    ax2.scatter(y_pred_test, residuals, alpha=0.6, s=80, c=residuals, 
               cmap='coolwarm', edgecolors='black', linewidth=0.5)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    
    ax2.set_xlabel(f'Predicted {target}', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Make prediction
    st.markdown("---")
    st.markdown("### 🎯 Make Your Own Prediction")
    st.write("Enter values for each feature to predict the target variable:")
    
    input_values = []
    
    # Create columns for inputs
    num_cols = min(len(features), 3)
    cols = st.columns(num_cols)
    
    for i, feat in enumerate(features):
        with cols[i % num_cols]:
            median_val = float(model_data[feat].median())
            min_val = float(model_data[feat].min())
            max_val = float(model_data[feat].max())
            
            val = st.number_input(
                f"📊 {feat}",
                value=median_val,
                min_value=min_val,
                max_value=max_val,
                format="%.2f",
                key=f"pred_{feat}",
                help=f"Enter a value between {min_val:.2f} and {max_val:.2f}"
            )
            input_values.append(val)
    
    # Predict button
    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
        prediction = model.predict(np.array(input_values).reshape(1, -1))[0]
        
        st.markdown("---")
        st.markdown("### 🎊 Prediction Result")
        
        # Display prediction with appropriate formatting
        if 'revenue' in target.lower() or 'gross' in target.lower() or 'budget' in target.lower():
            st.success(f"## 💰 Predicted {target}: ${prediction:,.2f}")
        elif 'score' in target.lower() or 'rating' in target.lower():
            st.success(f"## ⭐ Predicted {target}: {prediction:.2f}%")
        elif 'profitability' in target.lower():
            st.success(f"## 📊 Predicted {target}: {prediction:.2f}x")
        else:
            st.success(f"## 🎯 Predicted {target}: {prediction:.2f}")
        
        # Show input summary
        st.markdown("#### Input Values Used:")
        input_df = pd.DataFrame({
            'Feature': features,
            'Value': input_values
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)
        
        # Confidence interval for prediction
        st.markdown("#### 📊 Prediction Confidence")
        st.info(f"""
        Based on the model's performance:
        - **R² Score:** {r2_test:.4f}
        - **Average Error:** ±{mae_test:.2f}
        - **Confidence Range:** {prediction - mae_test:.2f} to {prediction + mae_test:.2f}
        
        This means the actual value is likely to be within ±{mae_test:.2f} of the predicted value.
        """)


elif page == "ℹ️ About Project":
    st.markdown("# ℹ️ Project Information")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
        <h2 style='color: white; text-align: center; margin: 0;'>🎬 Hollywood Movies Statistical Analysis System</h2>
        <p style='color: rgba(255,255,255,0.9); text-align: center; margin: 10px 0 0 0;'>Advanced Analytics & AI-Powered Predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 📊 Project Overview
    
    This is a comprehensive statistical analysis and machine learning system designed to analyze 
    Hollywood movie performance data. The application combines traditional statistical methods 
    with modern AI techniques to provide insights and predictions.
    
    ## 🎯 Key Features
    
    ### 1. 📈 Advanced Data Visualization
    - Interactive dashboards with real-time filtering
    - Multiple chart types (histograms, box plots, scatter plots, heatmaps)
    - Genre-based performance comparisons
    - Distribution analysis with KDE curves
    
    ### 2. 📉 Comprehensive Statistical Analysis
    - Descriptive statistics (mean, median, mode, variance, standard deviation)
    - Quartile analysis and IQR calculations
    - Skewness and kurtosis measurements
    - Data quality assessment
    
    ### 3. 🎯 Confidence Interval Estimation
    - Calculate confidence intervals for population parameters
    - Interactive confidence level adjustment (90-99%)
    - Visual representation of confidence regions
    - Margin of error calculations
    
    ### 4. 🎲 Probability Distribution Engine
    - **Normal Distribution Analysis:**
      - Distribution fitting with visual overlays
      - Probability calculations for any range
      - Shapiro-Wilk normality testing
      - Z-score computations
    
    - **Binomial Probability:**
      - Success probability calculations
      - Multiple probability types (exact, at least, at most)
      - Visual distribution charts
      - Real-world interpretations
    
    ### 5. 🔮 AI-Powered Predictions
    - Multiple linear regression modeling
    - Feature importance analysis
    - Model performance metrics (R², RMSE, MAE)
    - Residual analysis
    - Interactive prediction interface
    - Overfitting detection
    
    ## 📚 Dataset Information
    
    **Source:** [Hollywood Movies Dataset on Kaggle](https://www.kaggle.com/datasets/writuparnabanerjee/hollywood-movies)
    
    **Variables Include:**
    - Movie titles and metadata
    - Financial data (budget, revenue, profitability)
    - Ratings (audience scores, critics scores)
    - Categorical information (genre, studio)
    - Temporal data (release year)
    
    ## 🛠️ Technologies Used
    
    - **Language:** Python 3.14
    - **Framework:** Streamlit
    - **Data Analysis:** Pandas, NumPy
    - **Visualization:** Matplotlib, Seaborn, Plotly
    - **Statistics:** SciPy
    - **Machine Learning:** Scikit-learn
    
    ## 📈 Statistical Methods Implemented
    
    | Method | Application | Purpose |
    |--------|-------------|---------|
    | **Descriptive Statistics** | Summary measures | Understand data distribution |
    | **Confidence Intervals** | Parameter estimation | Estimate population parameters |
    | **Normal Distribution** | Probability modeling | Calculate probabilities |
    | **Binomial Distribution** | Success/failure events | Model binary outcomes |
    | **Linear Regression** | Prediction | Forecast movie performance |
    | **Correlation Analysis** | Relationship detection | Find variable associations |
    
    ## 👥 Development Team
    
    **Team Name:** [Data Detectives]
    
    | # | Roll Number |       Name                  | Department          | Section |
    |---|-------------|---------------------------------------------------------------  
    | 1 | [24F-0731]  |      [Muhammad Waleed]      | Computer Science    | BSCS-4D |
    | 2 | [24F-0717]  |      [Muhammad Mughees]     | Computer Science    | BSCS-4D |
    | 3 | [24F-0552]  |      [Faraz Ahmed]          | Computer Science    | BSCS-4D |
    | 4 | [24F-3050]  |      [Ibadat Ali]           | Software Engineering| BSSE-4A |
    | 5 | [24F-3113]  |      [Muhammad Adil]        | Software Engineering| BSSE-4B |
    
    ## 📅 Project Details
    
    - **Course:** Probability and Statistics
    - **Semester:** Spring 2026
    - **Submission Deadline:** May 7, 2026
    - **Institution:** [Your University Name]
    - **Instructor:** [Instructor Name]
    
    ## 🎓 Learning Outcomes
    
    Through this project, we have:
    
    ✅ Applied probability theory to real-world data  
    ✅ Implemented statistical inference techniques  
    ✅ Built predictive machine learning models  
    ✅ Created interactive data visualizations  
    ✅ Developed full-stack web applications  
    ✅ Gained experience with Python data science stack  
    
    ## 📊 Key Findings
    
    Our analysis revealed:
    
    - Strong correlation between budget and revenue
    - Genre significantly impacts movie profitability
    - Audience scores and critic scores show moderate correlation
    - Regression models can predict revenue with reasonable accuracy
    - Normal distribution fits well for most continuous variables
    
    ## 🔗 Resources
    
    - [Project GitHub Repository](https://github.com/waleedryaz/-Movie-Box-Office-Performance-Analyzer)
    - [Dataset Source](https://www.kaggle.com/datasets/writuparnabanerjee/hollywood-movies)
    - [Streamlit Documentation](https://docs.streamlit.io)
    - [Pandas Documentation](https://pandas.pydata.org)
    - [Scikit-learn Documentation](https://scikit-learn.org)
    
    ---
    
    <div style='text-align: center; padding: 20px; background-color: rgba(255,255,255,0.1); border-radius: 10px; margin-top: 30px;'>
        <p style='color: white; font-size: 16px; margin: 0;'>Made with ❤️ using Python & Streamlit</p>
        <p style='color: rgba(255,255,255,0.7); font-size: 12px; margin: 10px 0 0 0;'>© 2026 Hollywood Analytics Team</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #a0a0c0; font-size: 12px;'>
    <p style='margin: 5px 0;'>© 2026 Hollywood Analytics</p>
    <p style='margin: 5px 0;'>Powered by Streamlit</p>
    <p style='margin: 5px 0;'>Spring 2026</p>
</div>
""", unsafe_allow_html=True)
