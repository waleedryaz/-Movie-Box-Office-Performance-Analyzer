import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(
    page_title="Hollywood Movies Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {color: #ff4b4b;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('hollywood_movies.csv')
        
        # Clean all column names - remove extra spaces
        df.columns = df.columns.str.strip()
        
        # Display original columns for debugging
        st.sidebar.info(f"Columns found: {', '.join(df.columns[:3])}...")
        
        # Auto-detect numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Convert string numbers to numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                temp = df[col].astype(str).str.replace('$', '', regex=False)
                temp = temp.str.replace(',', '', regex=False)
                temp = temp.str.replace('%', '', regex=False)
                temp = pd.to_numeric(temp, errors='ignore')
                
                if pd.api.types.is_numeric_dtype(temp):
                    df[col] = temp
        
        # Update numeric columns after conversion
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return df, numeric_cols
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), []

df, numeric_cols = load_data()

if df.empty:
    st.error("❌ Could not load data. Please check if hollywood_movies.csv is in the repository.")
    
    # Show debug info
    st.subheader("Debug Information")
    try:
        test_df = pd.read_csv('hollywood_movies.csv', nrows=5)
        st.write("**File exists! Here are the first few rows:**")
        st.dataframe(test_df)
        st.write("**Column names in file:**")
        st.write(list(test_df.columns))
    except Exception as e:
        st.error(f"Cannot read file: {e}")
    
    st.stop()

# Get column names dynamically
all_columns = df.columns.tolist()
text_columns = df.select_dtypes(include=['object']).columns.tolist()

# Try to identify key columns by common patterns
movie_col = None
year_col = None
genre_col = None
studio_col = None

for col in all_columns:
    col_lower = col.lower()
    if 'movie' in col_lower or 'film' in col_lower or 'title' in col_lower:
        movie_col = col
    elif 'year' in col_lower:
        year_col = col
    elif 'genre' in col_lower:
        genre_col = col
    elif 'studio' in col_lower:
        studio_col = col

st.sidebar.title("🎬 Hollywood Movies")
st.sidebar.markdown("### Statistical Analysis Tool")
st.sidebar.markdown("---")

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
st.sidebar.markdown("### 🔧 Filters")

filtered_df = df.copy()

if genre_col and genre_col in df.columns:
    genres = sorted([g for g in df[genre_col].dropna().unique() if str(g) != 'nan'])
    if genres:
        selected_genres = st.sidebar.multiselect("Select Genres", genres, default=genres)
        filtered_df = filtered_df[filtered_df[genre_col].isin(selected_genres)]

if year_col and year_col in df.columns:
    year_data = pd.to_numeric(df[year_col], errors='coerce').dropna()
    if len(year_data) > 0:
        min_year = int(year_data.min())
        max_year = int(year_data.max())
        year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
        filtered_df = filtered_df[
            (pd.to_numeric(filtered_df[year_col], errors='coerce') >= year_range[0]) & 
            (pd.to_numeric(filtered_df[year_col], errors='coerce') <= year_range[1])
        ]

st.sidebar.success(f"✅ {len(filtered_df)} movies")

# Re-calculate numeric columns for filtered data
filtered_numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown("# 🎬 Hollywood Movies Statistical Analysis")
    st.markdown("### Profitability & Performance Prediction System")
    st.markdown("**Probability & Statistics Project | Spring 2026**")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📽️ Total Movies", f"{len(filtered_df):,}")
    
    with col2:
        if len(filtered_numeric_cols) > 0:
            first_numeric = filtered_numeric_cols[0]
            avg_val = filtered_df[first_numeric].mean()
            st.metric(f"📊 Avg {first_numeric}", f"{avg_val:.2f}")
    
    with col3:
        if len(filtered_numeric_cols) > 1:
            second_numeric = filtered_numeric_cols[1]
            avg_val = filtered_df[second_numeric].mean()
            st.metric(f"📈 Avg {second_numeric}", f"{avg_val:.2f}")
    
    with col4:
        if len(filtered_numeric_cols) > 2:
            third_numeric = filtered_numeric_cols[2]
            total_val = filtered_df[third_numeric].sum()
            st.metric(f"💰 Total {third_numeric}", f"{total_val:.2f}")
    
    st.markdown("---")
    st.markdown("### 📋 Dataset Preview")
    
    display_cols = all_columns[:8]
    st.dataframe(filtered_df[display_cols].head(15), use_container_width=True, hide_index=True)
    
    if len(filtered_numeric_cols) >= 2:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🏆 Top 5 by {filtered_numeric_cols[0]}")
            top_data = filtered_df.nlargest(5, filtered_numeric_cols[0])
            display_cols_top = [movie_col] if movie_col else []
            display_cols_top.append(filtered_numeric_cols[0])
            if genre_col:
                display_cols_top.append(genre_col)
            st.dataframe(top_data[display_cols_top], use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown(f"### 💎 Top 5 by {filtered_numeric_cols[1]}")
            top_data = filtered_df.nlargest(5, filtered_numeric_cols[1])
            display_cols_top = [movie_col] if movie_col else []
            display_cols_top.append(filtered_numeric_cols[1])
            if genre_col:
                display_cols_top.append(genre_col)
            st.dataframe(top_data[display_cols_top], use_container_width=True, hide_index=True)

# ============================================================================
# DATA EXPLORER
# ============================================================================

elif page == "📊 Data Explorer":
    st.markdown("# 📊 Data Explorer")
    
    tab1, tab2, tab3 = st.tabs(["📋 Overview", "🔍 Details", "❌ Data Quality"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", filtered_df.shape[0])
        with col2:
            st.metric("Columns", filtered_df.shape[1])
        with col3:
            st.metric("Numeric Vars", len(filtered_numeric_cols))
        
        st.markdown("---")
        st.markdown("### Full Dataset")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### Column Information")
        
        col_info = pd.DataFrame({
            'Column Name': filtered_df.columns,
            'Data Type': filtered_df.dtypes.astype(str).values,
            'Non-Null': filtered_df.count().values,
            'Null': filtered_df.isnull().sum().values,
            'Unique': [filtered_df[col].nunique() for col in filtered_df.columns]
        })
        
        st.dataframe(col_info, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("### Missing Values Analysis")
        
        missing = pd.DataFrame({
            'Column': filtered_df.columns,
            'Missing': filtered_df.isnull().sum().values,
            '%': (filtered_df.isnull().sum().values / len(filtered_df) * 100).round(2)
        })
        
        missing = missing[missing['Missing'] > 0]
        
        if len(missing) == 0:
            st.success("✅ No missing values!")
        else:
            st.dataframe(missing, use_container_width=True, hide_index=True)

# ============================================================================
# VISUAL ANALYTICS
# ============================================================================

elif page == "📈 Visual Analytics":
    st.markdown("# 📈 Visual Analytics Dashboard")
    
    if len(filtered_numeric_cols) == 0:
        st.warning("⚠️ No numeric columns found for visualization")
        st.stop()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📦 Comparisons", "🔵 Relationships", "🔥 Correlations"])
    
    with tab1:
        st.markdown("### Distribution Analysis")
        col = st.selectbox("Select Variable", filtered_numeric_cols)
        data = filtered_df[col].dropna()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
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
        st.markdown("### Comparison Analysis")
        col = st.selectbox("Select Variable", filtered_numeric_cols, key='comp')
        
        if genre_col:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            genres = filtered_df[genre_col].dropna().unique()
            data_by_cat = [filtered_df[filtered_df[genre_col] == g][col].dropna() for g in genres]
            
            bp = ax.boxplot(data_by_cat, labels=genres, patch_artist=True)
            
            for patch in bp['boxes']:
                patch.set_facecolor('#ff4b4b')
                patch.set_alpha(0.7)
            
            ax.set_xlabel(genre_col, fontsize=12)
            ax.set_ylabel(col, fontsize=12)
            ax.set_title(f'{col} by {genre_col}', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No categorical column found for comparison")
    
    with tab3:
        st.markdown("### Relationship Analysis")
        
        if len(filtered_numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("X Variable", filtered_numeric_cols, key='x')
            with col2:
                y_var = st.selectbox("Y Variable", filtered_numeric_cols, index=1, key='y')
            
            scatter_data = filtered_df[[x_var, y_var]].dropna()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(scatter_data[x_var], scatter_data[y_var], alpha=0.6, s=50, color='#ff4b4b')
            ax.set_xlabel(x_var, fontsize=12)
            ax.set_ylabel(y_var, fontsize=12)
            ax.set_title(f'{x_var} vs {y_var}', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Need at least 2 numeric columns")
    
    with tab4:
        st.markdown("### Correlation Matrix")
        
        if len(filtered_numeric_cols) >= 2:
            corr_data = filtered_df[filtered_numeric_cols].dropna()
            corr = corr_data.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            ax.set_xticks(np.arange(len(corr.columns)))
            ax.set_yticks(np.arange(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr.columns)
            
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha="center", va="center", color="black")
            
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Need at least 2 numeric columns")

# ============================================================================
# STATISTICS
# ============================================================================

elif page == "📉 Statistics":
    st.markdown("# 📉 Descriptive Statistics")
    
    if len(filtered_numeric_cols) == 0:
        st.warning("No numeric columns available")
        st.stop()
    
    tab1, tab2 = st.tabs(["📊 Summary", "🔍 Detailed"])
    
    with tab1:
        st.markdown("### Summary Statistics")
        st.dataframe(filtered_df[filtered_numeric_cols].describe(), use_container_width=True)
    
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
    st.markdown("# 🎯 Confidence Intervals")
    
    if len(filtered_numeric_cols) == 0:
        st.warning("No numeric columns available")
        st.stop()
    
    col = st.selectbox("Select Variable", filtered_numeric_cols)
    confidence = st.slider("Confidence %", 90, 99, 95)
    
    data = filtered_df[col].dropna()
    mean = data.mean()
    sem = stats.sem(data)
    ci = stats.t.interval(confidence/100, len(data)-1, loc=mean, scale=sem)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean", f"{mean:.4f}")
    with col2:
        st.metric("Lower", f"{ci[0]:.4f}")
    with col3:
        st.metric("Upper", f"{ci[1]:.4f}")
    
    st.success(f"We are {confidence}% confident the true mean is between {ci[0]:.4f} and {ci[1]:.4f}")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axvline(mean, color='red', linewidth=3, label='Mean')
    ax.axvline(ci[0], color='green', linewidth=2, linestyle='--', label='Lower')
    ax.axvline(ci[1], color='blue', linewidth=2, linestyle='--', label='Upper')
    ax.axvspan(ci[0], ci[1], alpha=0.3, color='yellow')
    ax.set_xlabel(col)
    ax.set_title(f'{confidence}% Confidence Interval')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yticks([])
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# PROBABILITY
# ============================================================================

elif page == "🎲 Probability":
    st.markdown("# 🎲 Probability Distribution")
    
    if len(filtered_numeric_cols) == 0:
        st.warning("No numeric columns available")
        st.stop()
    
    tab1, tab2 = st.tabs(["📊 Normal", "🎯 Binomial"])
    
    with tab1:
        col = st.selectbox("Select Variable", filtered_numeric_cols)
        data = filtered_df[col].dropna()
        
        mean = data.mean()
        std = data.std()
        
        st.metric("Mean", f"{mean:.4f}")
        st.metric("Std Dev", f"{std:.4f}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        x = np.linspace(data.min(), data.max(), 200)
        ax.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=3)
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.set_title(f'Normal Distribution: {col}')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("### Calculate Probability")
        col1, col2 = st.columns(2)
        with col1:
            lower = st.number_input("Lower", value=float(data.min()))
        with col2:
            upper = st.number_input("Upper", value=float(data.mean()))
        
        prob = stats.norm.cdf((upper-mean)/std) - stats.norm.cdf((lower-mean)/std)
        st.success(f"P({lower:.2f} < X < {upper:.2f}) = {prob:.4f} ({prob*100:.2f}%)")
    
    with tab2:
        if len(filtered_numeric_cols) > 0:
            col = st.selectbox("Select Variable", filtered_numeric_cols, key='binom')
            threshold = st.slider("Define Success Threshold", 
                                float(filtered_df[col].min()), 
                                float(filtered_df[col].max()), 
                                float(filtered_df[col].median()))
            
            success_data = filtered_df[col].dropna()
            p = (success_data > threshold).mean()
            
            st.info(f"Success probability: {p:.4f} ({p*100:.2f}%)")
            
            n = st.number_input("Number of trials", 1, 100, 10)
            k = st.number_input("Number of successes", 0, int(n), min(5, int(n)))
            
            exact = stats.binom.pmf(k, n, p)
            at_most = stats.binom.cdf(k, n, p)
            at_least = 1 - stats.binom.cdf(k-1, n, p)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"P(X={k})", f"{exact:.4f}")
            with col2:
                st.metric(f"P(X≤{k})", f"{at_most:.4f}")
            with col3:
                st.metric(f"P(X≥{k})", f"{at_least:.4f}")

# ============================================================================
# PREDICTIONS
# ============================================================================

elif page == "🔮 Predictions":
    st.markdown("# 🔮 Regression & Predictions")
    
    if len(filtered_numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns")
        st.stop()
    
    target = st.selectbox("Target Variable", filtered_numeric_cols)
    available = [c for c in filtered_numeric_cols if c != target]
    features = st.multiselect("Independent Variables", available, default=available[:min(2, len(available))])
    
    if len(features) == 0:
        st.warning("Select at least one feature")
        st.stop()
    
    model_data = filtered_df[[target] + features].dropna()
    
    if len(model_data) < 10:
        st.error("Not enough data")
        st.stop()
    
    X = model_data[features]
    y = model_data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    st.markdown("### Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.2f}")
    
    st.markdown("### Regression Equation")
    eq = f"{target} = {model.intercept_:.2f}"
    for feat, coef in zip(features, model.coef_):
        eq += f" + {coef:.2f}×{feat}"
    st.code(eq)
    
    st.markdown("### Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, s=50, color='#ff4b4b')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    ax.set_xlabel(f'Actual {target}')
    ax.set_ylabel(f'Predicted {target}')
    ax.set_title('Actual vs Predicted')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("### Make Prediction")
    inputs = []
    cols = st.columns(len(features))
    for i, feat in enumerate(features):
        with cols[i]:
            val = st.number_input(feat, value=float(model_data[feat].median()))
            inputs.append(val)
    
    if st.button("Predict"):
        pred = model.predict(np.array(inputs).reshape(1, -1))[0]
        st.success(f"Predicted {target}: {pred:.2f}")

# ============================================================================
# ABOUT
# ============================================================================

elif page == "📝 About":
    st.markdown("# 📝 Project Documentation")
    
    st.markdown("""
    ## 🎬 Hollywood Movies Statistical Analysis
    
    **Dataset:** Hollywood Movies  
    **Source:** https://www.kaggle.com/datasets/writuparnabanerjee/hollywood-movies
    
    ### Methods Applied
    - Graphical representation
    - Descriptive statistics
    - Confidence intervals
    - Probability distributions
    - Linear regression
    
    ### Technologies
    Python, Streamlit, Pandas, NumPy, SciPy, Scikit-learn
    
    ### Team Members
    1. [Roll] | [Name] | CS | [Section]
    2. [Roll] | [Name] | CS | [Section]
    3. [Roll] | [Name] | CS | [Section]
    4. [Roll] | [Name] | [Dept] | [Section]
    5. [Roll] | [Name] | [Dept] | [Section]
    
    **Spring 2026**
    """)

st.sidebar.markdown("---")
st.sidebar.info("Spring 2026")
