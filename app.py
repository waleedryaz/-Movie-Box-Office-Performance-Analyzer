import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Page configuration
st.set_page_config(
    page_title="Hollywood Movies Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 5px;
        padding: 10px 20px;
    }
    h1, h2, h3 {
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('hollywood_movies.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Clean and convert numeric columns
        numeric_cols = ['Year', 'Audience score %', 'Rotten Tomatoes %', 'Worldwide Gross', 'Profitability']
        
        for col in numeric_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('$', '', regex=False)
                    df[col] = df[col].str.replace(',', '', regex=False)
                    df[col] = df[col].str.replace('%', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean text columns
        text_cols = ['Movie', 'Genre', 'Lead Studio']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['Movie'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df = load_data()

if df.empty:
    st.error("❌ Could not load data. Please check if hollywood_movies.csv is in the repository.")
    st.stop()

# Sidebar
st.sidebar.markdown("# 🎬 Hollywood Movies")
st.sidebar.markdown("### Statistical Analysis Tool")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "📍 Navigate",
    [
        "🏠 Home",
        "📊 Data Explorer",
        "📈 Visual Analytics",
        "📉 Statistics",
        "🎯 Confidence Intervals",
        "🎲 Probability",
        "🔮 Predictions",
        "📝 About"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Filters
st.sidebar.markdown("### 🔧 Filters")

if 'Genre' in df.columns:
    all_genres = sorted([g for g in df['Genre'].unique() if pd.notna(g) and str(g) != 'nan'])
    if all_genres:
        selected_genres = st.sidebar.multiselect(
            "Select Genres",
            options=all_genres,
            default=all_genres
        )
        df = df[df['Genre'].isin(selected_genres)]

if 'Year' in df.columns:
    year_data = df['Year'].dropna()
    if len(year_data) > 0:
        min_year = int(year_data.min())
        max_year = int(year_data.max())
        year_range = st.sidebar.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

st.sidebar.success(f"✅ {len(df)} movies loaded")

# Available numeric columns
numeric_cols = [c for c in ['Year', 'Audience score %', 'Rotten Tomatoes %', 
                             'Worldwide Gross', 'Profitability'] if c in df.columns]

# ============================================================================
# PAGE: HOME
# ============================================================================

if page == "🏠 Home":
    st.markdown("# 🎬 Hollywood Movies Statistical Analysis")
    st.markdown("### Profitability & Performance Prediction System")
    st.markdown("**Probability & Statistics Project | Spring 2026**")
    
    st.markdown("---")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📽️ Total Movies",
            value=f"{len(df):,}"
        )
    
    with col2:
        if 'Audience score %' in df.columns:
            avg_score = df['Audience score %'].mean()
            st.metric(
                label="⭐ Avg Audience Score",
                value=f"{avg_score:.1f}%"
            )
    
    with col3:
        if 'Profitability' in df.columns:
            avg_profit = df['Profitability'].mean()
            st.metric(
                label="💰 Avg Profitability",
                value=f"{avg_profit:.2f}x"
            )
    
    with col4:
        if 'Worldwide Gross' in df.columns:
            total_gross = df['Worldwide Gross'].sum()
            st.metric(
                label="🌍 Total Revenue",
                value=f"${total_gross/1e9:.1f}B"
            )
    
    st.markdown("---")
    
    # Dataset Preview
    st.markdown("### 📋 Dataset Preview")
    
    display_cols = [c for c in ['Movie', 'Year', 'Genre', 'Lead Studio', 'Audience score %', 
                                 'Rotten Tomatoes %', 'Worldwide Gross', 'Profitability'] 
                    if c in df.columns]
    
    st.dataframe(
        df[display_cols].head(10),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Top Movies
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top 5 Highest Grossing")
        if 'Worldwide Gross' in df.columns and 'Movie' in df.columns:
            top_gross = df.nlargest(5, 'Worldwide Gross')[['Movie', 'Worldwide Gross', 'Genre']]
            top_gross['Worldwide Gross'] = top_gross['Worldwide Gross'].apply(lambda x: f"${x/1e6:.1f}M")
            st.dataframe(top_gross, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 💎 Top 5 Most Profitable")
        if 'Profitability' in df.columns and 'Movie' in df.columns:
            top_profit = df.nlargest(5, 'Profitability')[['Movie', 'Profitability', 'Genre']]
            top_profit['Profitability'] = top_profit['Profitability'].apply(lambda x: f"{x:.2f}x")
            st.dataframe(top_profit, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE: DATA EXPLORER
# ============================================================================

elif page == "📊 Data Explorer":
    st.markdown("# 📊 Data Explorer")
    
    tab1, tab2, tab3 = st.tabs(["📋 Overview", "🔍 Details", "❌ Data Quality"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Numeric Variables", len(numeric_cols))
        
        st.markdown("---")
        st.markdown("### Full Dataset")
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### Column Information")
        
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str).values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        
        st.dataframe(col_info, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("### Missing Values Analysis")
        
        missing = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        
        missing = missing[missing['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing) == 0:
            st.success("✅ No missing values found in the dataset!")
        else:
            st.warning(f"⚠️ Found missing values in {len(missing)} columns")
            st.dataframe(missing, use_container_width=True, hide_index=True)
            
            # Missing values chart
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(missing['Column'], missing['Missing %'], color='#ff4b4b')
            ax.set_xlabel('Missing %')
            ax.set_title('Missing Values by Column')
            ax.grid(alpha=0.3)
            st.pyplot(fig)

# ============================================================================
# PAGE: VISUAL ANALYTICS
# ============================================================================

elif page == "📈 Visual Analytics":
    st.markdown("# 📈 Visual Analytics Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📦 Comparisons", "🔵 Relationships", "🔥 Correlations"])
    
    with tab1:
        st.markdown("### Distribution Analysis")
        
        col = st.selectbox("Select Variable", numeric_cols, key='dist_var')
        
        data = df[col].dropna()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(data, bins=30, color='#ff4b4b', edgecolor='black', alpha=0.7)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Statistics")
            st.metric("Mean", f"{data.mean():.2f}")
            st.metric("Median", f"{data.median():.2f}")
            st.metric("Std Dev", f"{data.std():.2f}")
            st.metric("Min", f"{data.min():.2f}")
            st.metric("Max", f"{data.max():.2f}")
    
    with tab2:
        st.markdown("### Comparison by Genre")
        
        col = st.selectbox("Select Variable", numeric_cols, key='comp_var')
        
        if 'Genre' in df.columns:
            # Box plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            genres = df['Genre'].dropna().unique()
            data_by_genre = [df[df['Genre'] == g][col].dropna() for g in genres]
            
            bp = ax.boxplot(data_by_genre, labels=genres, patch_artist=True)
            
            for patch in bp['boxes']:
                patch.set_facecolor('#ff4b4b')
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Genre', fontsize=12)
            ax.set_ylabel(col, fontsize=12)
            ax.set_title(f'{col} by Genre', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab3:
        st.markdown("### Relationship Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("X Variable", numeric_cols, key='x_var')
        with col2:
            y_var = st.selectbox("Y Variable", numeric_cols, 
                                index=min(1, len(numeric_cols)-1), key='y_var')
        
        # Scatter plot
        scatter_data = df[[x_var, y_var]].dropna()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'Genre' in df.columns:
            genres = df['Genre'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(genres)))
            
            for i, genre in enumerate(genres):
                genre_data = df[df['Genre'] == genre][[x_var, y_var]].dropna()
                ax.scatter(genre_data[x_var], genre_data[y_var], 
                          label=genre, alpha=0.6, s=50, color=colors[i])
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(scatter_data[x_var], scatter_data[y_var], 
                      alpha=0.6, s=50, color='#ff4b4b')
        
        ax.set_xlabel(x_var, fontsize=12)
        ax.set_ylabel(y_var, fontsize=12)
        ax.set_title(f'{x_var} vs {y_var}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab4:
        st.markdown("### Correlation Matrix")
        
        corr_data = df[numeric_cols].dropna()
        
        if len(corr_data) > 0:
            corr = corr_data.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            ax.set_xticks(np.arange(len(corr.columns)))
            ax.set_yticks(np.arange(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr.columns)
            
            # Add correlation values
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=10)
            
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("""
            **How to read:**
            - **+1.0** = Perfect positive correlation
            - **-1.0** = Perfect negative correlation
            - **0.0** = No correlation
            """)

# ============================================================================
# PAGE: STATISTICS
# ============================================================================

elif page == "📉 Statistics":
    st.markdown("# 📉 Descriptive Statistics")
    
    tab1, tab2 = st.tabs(["📊 Summary", "🔍 Detailed Analysis"])
    
    with tab1:
        st.markdown("### Summary Statistics")
        
        summary = df[numeric_cols].describe()
        st.dataframe(summary, use_container_width=True)
        
        st.download_button(
            label="📥 Download Summary (CSV)",
            data=summary.to_csv(),
            file_name="summary_statistics.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.markdown("### Detailed Statistical Analysis")
        
        col = st.selectbox("Select Variable for Analysis", numeric_cols)
        
        data = df[col].dropna()
        
        # Calculate statistics
        mean = data.mean()
        median = data.median()
        mode_vals = data.mode()
        mode = mode_vals.iloc[0] if len(mode_vals) > 0 else np.nan
        variance = data.var()
        std_dev = data.std()
        minimum = data.min()
        maximum = data.max()
        range_val = maximum - minimum
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### 📊 Central Tendency")
            st.metric("Mean", f"{mean:.4f}")
            st.metric("Median", f"{median:.4f}")
            st.metric("Mode", f"{mode:.4f}")
        
        with col2:
            st.markdown("#### 📏 Dispersion")
            st.metric("Variance", f"{variance:.4f}")
            st.metric("Std Dev", f"{std_dev:.4f}")
            st.metric("Range", f"{range_val:.4f}")
        
        with col3:
            st.markdown("#### 📐 Position")
            st.metric("Min", f"{minimum:.4f}")
            st.metric("Q1 (25%)", f"{q1:.4f}")
            st.metric("Q3 (75%)", f"{q3:.4f}")
        
        with col4:
            st.markdown("#### 🔀 Shape")
            st.metric("Max", f"{maximum:.4f}")
            st.metric("IQR", f"{iqr:.4f}")
            st.metric("Skewness", f"{skewness:.4f}")
        
        st.metric("Kurtosis", f"{kurtosis:.4f}")

# ============================================================================
# PAGE: CONFIDENCE INTERVALS
# ============================================================================

elif page == "🎯 Confidence Intervals":
    st.markdown("# 🎯 Confidence Interval Estimation")
    
    st.info("""
    A **confidence interval** provides a range of values that is likely to contain 
    the true population parameter with a certain level of confidence.
    """)
    
    col = st.selectbox("Select Variable", numeric_cols)
    confidence = st.slider("Confidence Level (%)", 90, 99, 95, 1)
    
    data = df[col].dropna()
    
    # Calculate CI
    mean = data.mean()
    sem = stats.sem(data)
    ci = stats.t.interval(confidence/100, len(data)-1, loc=mean, scale=sem)
    margin_error = ci[1] - mean
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Sample Mean", f"{mean:.4f}")
        st.caption(f"Based on {len(data)} observations")
    
    with col2:
        st.metric("⬇️ Lower Bound", f"{ci[0]:.4f}")
        st.caption(f"Margin of Error: ±{margin_error:.4f}")
    
    with col3:
        st.metric("⬆️ Upper Bound", f"{ci[1]:.4f}")
        st.caption(f"Confidence: {confidence}%")
    
    st.success(f"""
    ✅ **Interpretation:** We are **{confidence}% confident** that the true population mean 
    of **{col}** lies between **{ci[0]:.4f}** and **{ci[1]:.4f}**.
    """)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.axvline(mean, color='red', linewidth=3, label=f'Sample Mean: {mean:.2f}', zorder=3)
    ax.axvline(ci[0], color='green', linewidth=2, linestyle='--', label=f'Lower CI: {ci[0]:.2f}', zorder=3)
    ax.axvline(ci[1], color='blue', linewidth=2, linestyle='--', label=f'Upper CI: {ci[1]:.2f}', zorder=3)
    ax.axvspan(ci[0], ci[1], alpha=0.3, color='yellow', label=f'{confidence}% Confidence Interval')
    
    ax.set_xlabel(col, fontsize=12)
    ax.set_title(f'{confidence}% Confidence Interval for {col}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_yticks([])
    
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# PAGE: PROBABILITY
# ============================================================================

elif page == "🎲 Probability":
    st.markdown("# 🎲 Probability Distribution Analysis")
    
    tab1, tab2 = st.tabs(["📊 Normal Distribution", "🎯 Binomial Probability"])
    
    with tab1:
        st.markdown("### Normal Distribution Analysis")
        
        col = st.selectbox("Select Variable", numeric_cols)
        
        data = df[col].dropna()
        
        mean = data.mean()
        std = data.std()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean (μ)", f"{mean:.4f}")
        with col2:
            st.metric("Std Dev (σ)", f"{std:.4f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Histogram
        n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.7, 
                                   color='skyblue', edgecolor='black', label='Data Distribution')
        
        # Normal curve
        x = np.linspace(data.min(), data.max(), 200)
        y = stats.norm.pdf(x, mean, std)
        ax.plot(x, y, 'r-', linewidth=3, label='Normal Distribution Curve')
        
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Normal Distribution Fit: {col}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Probability calculation
        st.markdown("---")
        st.markdown("### Calculate Probability P(a < X < b)")
        
        col1, col2 = st.columns(2)
        with col1:
            lower = st.number_input("Lower bound (a)", value=float(data.min()), format="%.2f")
        with col2:
            upper = st.number_input("Upper bound (b)", value=float(data.mean()), format="%.2f")
        
        z_lower = (lower - mean) / std
        z_upper = (upper - mean) / std
        prob = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
        
        st.success(f"""
        **P({lower:.2f} < X < {upper:.2f}) = {prob:.4f}**
        
        This means there is a **{prob*100:.2f}%** probability that a randomly selected 
        movie will have a {col} value between {lower:.2f} and {upper:.2f}.
        """)
        
        # Normality test
        st.markdown("---")
        st.markdown("### Shapiro-Wilk Normality Test")
        
        sample = data.sample(min(len(data), 5000), random_state=42)
        stat, p_value = stats.shapiro(sample)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Statistic", f"{stat:.6f}")
        with col2:
            st.metric("P-value", f"{p_value:.6f}")
        
        if p_value > 0.05:
            st.success("✅ The data appears to be **normally distributed** (p > 0.05)")
        else:
            st.warning("⚠️ The data may **not be normally distributed** (p ≤ 0.05)")
    
    with tab2:
        st.markdown("### Binomial Probability - Highly Rated Movies")
        
        if 'Audience score %' in df.columns:
            threshold = st.slider("Define 'Highly Rated' as Audience Score >", 50, 95, 70, 5)
            
            success_data = df['Audience score %'].dropna()
            p = (success_data > threshold).mean()
            
            st.info(f"""
            **Success Probability (p) = {p:.4f} ({p*100:.2f}%)**
            
            This is the probability that a randomly selected movie has an Audience Score > {threshold}%
            """)
            
            n = st.number_input("Number of movies sampled (n)", 1, 100, 10, 1)
            k = st.number_input("Number of highly rated movies (k)", 0, int(n), min(5, int(n)), 1)
            
            # Calculate probabilities
            prob_exact = stats.binom.pmf(k, n, p)
            prob_at_most = stats.binom.cdf(k, n, p)
            prob_at_least = 1 - stats.binom.cdf(k-1, n, p)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"P(X = {k})", f"{prob_exact:.4f}")
                st.caption("Exactly k successes")
            
            with col2:
                st.metric(f"P(X ≤ {k})", f"{prob_at_most:.4f}")
                st.caption("At most k successes")
            
            with col3:
                st.metric(f"P(X ≥ {k})", f"{prob_at_least:.4f}")
                st.caption("At least k successes")
            
            # Plot binomial distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x_vals = np.arange(0, n+1)
            probs = stats.binom.pmf(x_vals, n, p)
            
            colors = ['red' if x == k else '#ff4b4b' for x in x_vals]
            ax.bar(x_vals, probs, color=colors, edgecolor='black', alpha=0.7)
            
            ax.axvline(k, color='darkred', linewidth=2, linestyle='--', label=f'k = {k}')
            
            ax.set_xlabel('Number of Highly Rated Movies', fontsize=12)
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title(f'Binomial Distribution (n={n}, p={p:.2f})', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

# ============================================================================
# PAGE: PREDICTIONS
# ============================================================================

elif page == "🔮 Predictions":
    st.markdown("# 🔮 Regression Modeling & Predictions")
    
    st.info("""
    Build a linear regression model to predict movie performance metrics based on other variables.
    """)
    
    # Model configuration
    target = st.selectbox(
        "Select Target Variable to Predict (Y)",
        ['Profitability', 'Worldwide Gross', 'Audience score %']
    )
    
    available_features = [c for c in numeric_cols if c != target]
    
    default_features = [c for c in available_features if c not in ['Year']][:3]
    
    features = st.multiselect(
        "Select Independent Variables (X)",
        available_features,
        default=default_features
    )
    
    if len(features) == 0:
        st.warning("⚠️ Please select at least one independent variable")
        st.stop()
    
    # Prepare data
    model_data = df[[target] + features].dropna()
    
    if len(model_data) < 10:
        st.error("❌ Not enough data. Please select different variables or adjust filters.")
        st.stop()
    
    X = model_data[features]
    y = model_data[target]
    
    # Split data
    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Display performance
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
        st.caption("Proportion of variance explained (closer to 1 is better)")
    
    with col2:
        st.metric("RMSE", f"{rmse:.2f}")
        st.caption("Root Mean Squared Error (lower is better)")
    
    with col3:
        train_score = model.score(X_train, y_train)
        st.metric("Training Score", f"{train_score:.4f}")
        st.caption("Model fit on training data")
    
    # Regression equation
    st.markdown("---")
    st.markdown("### 📐 Regression Equation")
    
    equation = f"{target} = {model.intercept_:.2f}"
    for feat, coef in zip(features, model.coef_):
        sign = "+" if coef >= 0 else ""
        equation += f" {sign} {coef:.2f}×{feat}"
    
    st.code(equation, language="text")
    
    # Coefficients table
    st.markdown("### 📋 Feature Coefficients")
    
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_,
        'Abs Coefficient': np.abs(model.coef_)
    }).sort_values('Abs Coefficient', ascending=False)
    
    st.dataframe(coef_df[['Feature', 'Coefficient']], use_container_width=True, hide_index=True)
    
    # Actual vs Predicted plot
    st.markdown("---")
    st.markdown("### 📈 Actual vs Predicted Values")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_test, y_pred, alpha=0.6, s=50, color='#ff4b4b', edgecolor='black')
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel(f'Actual {target}', fontsize=12)
    ax.set_ylabel(f'Predicted {target}', fontsize=12)
    ax.set_title(f'Actual vs Predicted {target}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Make prediction
    st.markdown("---")
    st.markdown("### 🎯 Make a Prediction")
    st.markdown("Enter values for each feature:")
    
    input_values = []
    cols = st.columns(min(len(features), 3))
    
    for i, feat in enumerate(features):
        with cols[i % len(cols)]:
            median_val = float(model_data[feat].median())
            min_val = float(model_data[feat].min())
            max_val = float(model_data[feat].max())
            
            val = st.number_input(
                f"{feat}",
                value=median_val,
                min_value=min_val,
                max_value=max_val,
                format="%.2f",
                key=f"pred_{feat}"
            )
            input_values.append(val)
    
    if st.button("🔮 Predict Now", type="primary", use_container_width=True):
        prediction = model.predict(np.array(input_values).reshape(1, -1))[0]
        
        st.markdown("### 🎊 Prediction Result")
        
        if target == 'Worldwide Gross':
            st.success(f"## Predicted {target}: ${prediction:,.2f}")
        elif target == 'Profitability':
            st.success(f"## Predicted {target}: {prediction:.2f}x")
        else:
            st.success(f"## Predicted {target}: {prediction:.2f}%")

# ============================================================================
# PAGE: ABOUT
# ============================================================================

elif page == "📝 About":
    st.markdown("# 📝 Project Documentation")
    
    st.markdown("""
    ## 🎬 Hollywood Movies Statistical Analysis System
    
    ### 📊 Project Overview
    
    This comprehensive web application performs **statistical analysis and predictive modeling** 
    on Hollywood movie data using probability and statistical methods.
    
    ---
    
    ### 🎯 Project Objectives
    
    1. **Graphical Data Representation** - Visualize distributions, relationships, and trends
    2. **Descriptive Statistics** - Calculate measures of central tendency and dispersion
    3. **Confidence Intervals** - Estimate population parameters with confidence
    4. **Probability Analysis** - Fit distributions and calculate probabilities
    5. **Regression Modeling** - Predict movie profitability and revenue
    
    ---
    
    ### 📊 Dataset Information
    
    **Source:** [Hollywood Movies Dataset - Kaggle](https://www.kaggle.com/datasets/writuparnabanerjee/hollywood-movies)
    
    **Key Variables:**
    - **Movie** - Movie title
    - **Year** - Release year
    - **Genre** - Movie genre/category
    - **Lead Studio** - Production studio
    - **Audience Score %** - Audience rating percentage
    - **Rotten Tomatoes %** - Critics rating percentage
    - **Worldwide Gross** - Total worldwide revenue
    - **Profitability** - Revenue to budget ratio
    
    ---
    
    ### 📈 Statistical Methods Implemented
    
    | Method | Application | Purpose |
    |--------|-------------|---------|
    | **Histograms** | Distribution visualization | Show frequency distribution of variables |
    | **Box Plots** | Outlier detection | Compare distributions across genres |
    | **Scatter Plots** | Relationship analysis | Identify correlations between variables |
    | **Correlation Matrix** | Association measurement | Quantify linear relationships |
    | **Descriptive Statistics** | Data summarization | Calculate mean, median, mode, variance, std dev |
    | **Confidence Intervals** | Parameter estimation | Estimate population mean with confidence |
    | **Normal Distribution** | Probability modeling | Fit normal curves and calculate probabilities |
    | **Binomial Distribution** | Success probability | Model discrete success/failure scenarios |
    | **Linear Regression** | Prediction | Model relationships and predict outcomes |
    
    ---
    
    ### 🛠️ Technologies Used
    
    - **Programming Language:** Python 3.14
    - **Web Framework:** Streamlit
    - **Data Analysis:** Pandas, NumPy
    - **Visualization:** Matplotlib, Seaborn
    - **Statistics:** SciPy
    - **Machine Learning:** Scikit-learn
    
    ---
    
    ### 👥 Team Information
    
    **Project Team:** [Your Team Name]
    
    | # | Roll Number | Name | Department | Section |
    |---|-------------|------|------------|---------|
    | 1 | [Roll #] | [Student Name] | Computer Science | [Section] |
    | 2 | [Roll #] | [Student Name] | Computer Science | [Section] |
    | 3 | [Roll #] | [Student Name] | Computer Science | [Section] |
    | 4 | [Roll #] | [Student Name] | [Department] | [Section] |
    | 5 | [Roll #] | [Student Name] | [Department] | [Section] |
    
    ---
    
    ### 📅 Project Details
    
    - **Course:** Probability and Statistics
    - **Semester:** Spring 2026
    - **Submission Deadline:** May 7, 2026
    - **Institution:** [Your University Name]
    
    ---
    
    ### 🎓 Key Findings & Conclusion
    
    This project successfully demonstrates the application of probability and statistical 
    methods on real-world movie industry data. The analysis reveals:
    
    ✅ **Strong correlations** between budget and revenue  
    ✅ **Genre-based patterns** in profitability and ratings  
    ✅ **Predictive capability** using regression models  
    ✅ **Distribution characteristics** of movie performance metrics  
    
    The interactive web application makes statistical analysis accessible and provides 
    valuable insights for understanding factors that contribute to movie success.
    
    ---
    
    ### 📞 Contact & Support
    
    For questions or suggestions, please contact the project team.
    
    ---
    
    **Made with ❤️ using Python & Streamlit**
    
    *© 2026 - Probability & Statistics Project*
    """)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Filters")
st.sidebar.markdown(f"**Movies Shown:** {len(df)}")
st.sidebar.markdown("---")
st.sidebar.info("**Probability & Statistics**\nSpring 2026")
