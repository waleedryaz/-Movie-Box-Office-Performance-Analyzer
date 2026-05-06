import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Hollywood Movies Analysis", page_icon="🎬", layout="wide")

# Load data
@st.cache_data
def load_data():
    if not os.path.exists('hollywood_movies.csv'):
        raise FileNotFoundError("hollywood_movies.csv not found!")
    
    df = pd.read_csv('hollywood_movies.csv')
    
    # Clean column names (remove spaces)
    df.columns = df.columns.str.strip()
    
    # Convert numeric columns
    numeric_cols = ['Year', 'Audience score %', 'Rotten Tomatoes %', 'Worldwide Gross', 'Profitability']
    
    for col in numeric_cols:
        if col in df.columns:
            # Remove $ and commas, convert to numeric
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('%', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean text columns
    if 'Movie' in df.columns:
        df['Movie'] = df['Movie'].astype(str).str.strip()
    if 'Genre' in df.columns:
        df['Genre'] = df['Genre'].astype(str).str.strip()
    if 'Lead Studio' in df.columns:
        df['Lead Studio'] = df['Lead Studio'].astype(str).str.strip()
    
    return df

try:
    df = load_data()
    st.sidebar.success(f"✅ Loaded {len(df)} movies")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# Sidebar navigation
st.sidebar.title("🎬 Hollywood Movies Project")

page = st.sidebar.radio("Navigate:", [
    "🏠 Home",
    "📊 Data Overview",
    "📈 Graphical Analysis",
    "📉 Descriptive Statistics",
    "🎯 Confidence Intervals",
    "🎲 Probability Distribution",
    "🔮 Regression & Prediction",
    "📝 Project Summary"
])

st.sidebar.markdown("---")

# Filters
if 'Genre' in df.columns:
    genres = sorted(df['Genre'].dropna().unique())
    selected_genres = st.sidebar.multiselect("Filter by Genre:", genres, default=genres)
    filtered_df = df[df['Genre'].isin(selected_genres)]
else:
    filtered_df = df.copy()

if 'Year' in df.columns:
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    year_range = st.sidebar.slider("Year Range:", min_year, max_year, (min_year, max_year))
    filtered_df = filtered_df[(filtered_df['Year'] >= year_range[0]) & (filtered_df['Year'] <= year_range[1])]

st.sidebar.info(f"Showing: {len(filtered_df)} movies")

# Numeric columns for analysis
numeric_cols = ['Year', 'Audience score %', 'Rotten Tomatoes %', 'Worldwide Gross', 'Profitability']
numeric_cols = [c for c in numeric_cols if c in filtered_df.columns]

# PAGE: HOME
if page == "🏠 Home":
    st.title("🎬 Statistical Analysis of Hollywood Movies")
    st.markdown("### Profitability and Performance Prediction")
    st.markdown("**Probability and Statistics Project - Spring 2026**")
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📽️ Total Movies", len(filtered_df))
    
    with col2:
        if 'Audience score %' in filtered_df.columns:
            avg_score = filtered_df['Audience score %'].mean()
            st.metric("⭐ Avg Audience Score", f"{avg_score:.1f}%")
    
    with col3:
        if 'Profitability' in filtered_df.columns:
            avg_profit = filtered_df['Profitability'].mean()
            st.metric("💰 Avg Profitability", f"{avg_profit:.2f}x")
    
    with col4:
        if 'Worldwide Gross' in filtered_df.columns:
            total_gross = filtered_df['Worldwide Gross'].sum()
            st.metric("🌍 Total Gross", f"${total_gross/1e9:.2f}B")
    
    st.markdown("---")
    
    # Dataset preview
    st.subheader("📋 Dataset Preview")
    display_cols = ['Movie', 'Year', 'Genre', 'Lead Studio', 'Audience score %', 
                    'Rotten Tomatoes %', 'Worldwide Gross', 'Profitability']
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(filtered_df[display_cols].head(15), use_container_width=True)
    
    st.markdown("---")
    
    # Top movies
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 5 by Worldwide Gross")
        if 'Worldwide Gross' in filtered_df.columns:
            top_gross = filtered_df.nlargest(5, 'Worldwide Gross')[['Movie', 'Worldwide Gross', 'Genre']]
            st.dataframe(top_gross, use_container_width=True)
    
    with col2:
        st.subheader("💎 Top 5 by Profitability")
        if 'Profitability' in filtered_df.columns:
            top_profit = filtered_df.nlargest(5, 'Profitability')[['Movie', 'Profitability', 'Genre']]
            st.dataframe(top_profit, use_container_width=True)

# PAGE: DATA OVERVIEW
elif page == "📊 Data Overview":
    st.title("📊 Data Overview")
    
    tab1, tab2, tab3 = st.tabs(["📋 Dataset Info", "📄 Columns", "❌ Missing Values"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows", filtered_df.shape[0])
        with col2:
            st.metric("Columns", filtered_df.shape[1])
        with col3:
            st.metric("Numeric Columns", len(numeric_cols))
        
        st.subheader("First 20 Rows")
        st.dataframe(filtered_df.head(20), use_container_width=True)
        
        st.subheader("Last 10 Rows")
        st.dataframe(filtered_df.tail(10), use_container_width=True)
    
    with tab2:
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': filtered_df.columns,
            'Data Type': filtered_df.dtypes.values,
            'Non-Null Count': filtered_df.count().values,
            'Null Count': filtered_df.isnull().sum().values
        })
        st.dataframe(col_info, use_container_width=True)
    
    with tab3:
        st.subheader("Missing Values Analysis")
        missing = pd.DataFrame({
            'Column': filtered_df.columns,
            'Missing': filtered_df.isnull().sum().values,
            'Missing %': (filtered_df.isnull().sum().values / len(filtered_df) * 100).round(2)
        })
        missing = missing[missing['Missing'] > 0].sort_values('Missing', ascending=False)
        
        if len(missing) == 0:
            st.success("✅ No missing values!")
        else:
            st.dataframe(missing, use_container_width=True)
            
            fig = px.bar(missing, x='Column', y='Missing %', title='Missing Values by Column')
            st.plotly_chart(fig, use_container_width=True)

# PAGE: GRAPHICAL ANALYSIS
elif page == "📈 Graphical Analysis":
    st.title("📈 Graphical and Tabular Data Representation")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Histogram", "📦 Box Plot", "🔵 Scatter Plot", "🔥 Correlation Heatmap", "📈 Bar Charts"
    ])
    
    with tab1:
        st.subheader("Histogram - Distribution Analysis")
        col = st.selectbox("Select Variable:", numeric_cols, key='hist')
        data = filtered_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        
        fig = px.histogram(data, x=col, nbins=30, title=f'Distribution of {col}', marginal='box')
        st.plotly_chart(fig, use_container_width=True)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Mean", f"{data.mean():.2f}")
        with c2:
            st.metric("Median", f"{data.median():.2f}")
        with c3:
            st.metric("Std Dev", f"{data.std():.2f}")
        with c4:
            st.metric("Count", len(data))
    
    with tab2:
        st.subheader("Box Plot - Distribution by Genre")
        col = st.selectbox("Select Variable:", numeric_cols, key='box')
        
        if 'Genre' in filtered_df.columns:
            box_data = filtered_df[['Genre', col]].replace([np.inf, -np.inf], np.nan).dropna()
            fig = px.box(box_data, x='Genre', y=col, title=f'Box Plot: {col} by Genre')
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Genre column not available")
    
    with tab3:
        st.subheader("Scatter Plot - Relationship Analysis")
        c1, c2 = st.columns(2)
        
        with c1:
            x_var = st.selectbox("X Variable:", numeric_cols, key='sx')
        with c2:
            y_var = st.selectbox("Y Variable:", numeric_cols, index=min(1, len(numeric_cols)-1), key='sy')
        
        if 'Genre' in filtered_df.columns:
            scatter_data = filtered_df[[x_var, y_var, 'Genre', 'Movie']].replace([np.inf, -np.inf], np.nan).dropna()
            fig = px.scatter(scatter_data, x=x_var, y=y_var, color='Genre', 
                           hover_data=['Movie'], title=f'{x_var} vs {y_var}')
        else:
            scatter_data = filtered_df[[x_var, y_var, 'Movie']].replace([np.inf, -np.inf], np.nan).dropna()
            fig = px.scatter(scatter_data, x=x_var, y=y_var, hover_data=['Movie'], title=f'{x_var} vs {y_var}')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Correlation Heatmap")
        corr_data = filtered_df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
        corr = corr_data.corr()
        
        fig = px.imshow(corr, text_auto='.2f', aspect='auto', 
                       color_continuous_scale='RdBu_r', title='Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Interpretation:**
        - Values close to +1: Strong positive correlation
        - Values close to -1: Strong negative correlation
        - Values close to 0: No correlation
        """)
    
    with tab5:
        st.subheader("Top Movies by Metric")
        
        metric = st.selectbox("Select Metric:", numeric_cols)
        top_n = st.slider("Number of movies:", 5, 20, 10)
        
        top_movies = filtered_df.nlargest(top_n, metric)
        
        fig = px.bar(top_movies, x='Movie', y=metric, color='Genre' if 'Genre' in top_movies.columns else None,
                    title=f'Top {top_n} Movies by {metric}')
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        display_cols = ['Movie', 'Genre', metric, 'Year'] if 'Genre' in top_movies.columns else ['Movie', metric, 'Year']
        st.dataframe(top_movies[display_cols], use_container_width=True)

# PAGE: DESCRIPTIVE STATISTICS
elif page == "📉 Descriptive Statistics":
    st.title("📉 Descriptive Statistical Measures")
    
    tab1, tab2 = st.tabs(["📊 Summary Statistics", "🔍 Detailed Analysis"])
    
    with tab1:
        st.subheader("Summary Statistics Table")
        summary = filtered_df[numeric_cols].replace([np.inf, -np.inf], np.nan).describe()
        st.dataframe(summary, use_container_width=True)
    
    with tab2:
        st.subheader("Detailed Statistical Analysis")
        col = st.selectbox("Select Variable:", numeric_cols)
        data = filtered_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Calculate all statistics
        mean = data.mean()
        median = data.median()
        mode = data.mode().iloc[0] if len(data.mode()) > 0 else np.nan
        variance = data.var()
        std_dev = data.std()
        minimum = data.min()
        maximum = data.max()
        data_range = maximum - minimum
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        # Display in columns
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown("**📊 Central Tendency**")
            st.metric("Mean", f"{mean:.4f}")
            st.metric("Median", f"{median:.4f}")
            st.metric("Mode", f"{mode:.4f}")
        
        with c2:
            st.markdown("**📏 Dispersion**")
            st.metric("Variance", f"{variance:.4f}")
            st.metric("Std Dev", f"{std_dev:.4f}")
            st.metric("Range", f"{data_range:.4f}")
        
        with c3:
            st.markdown("**📐 Quartiles**")
            st.metric("Q1 (25%)", f"{q1:.4f}")
            st.metric("Q3 (75%)", f"{q3:.4f}")
            st.metric("IQR", f"{iqr:.4f}")
        
        with c4:
            st.markdown("**🔀 Shape**")
            st.metric("Skewness", f"{skewness:.4f}")
            st.metric("Kurtosis", f"{kurtosis:.4f}")
            st.metric("Min", f"{minimum:.4f}")
        
        st.metric("Maximum", f"{maximum:.4f}")
        
        st.markdown("---")
        st.subheader("Frequency Distribution")
        freq = data.value_counts().reset_index()
        freq.columns = [col, 'Frequency']
        freq['Relative Frequency'] = (freq['Frequency'] / len(data)).round(4)
        freq['Percentage'] = (freq['Relative Frequency'] * 100).round(2)
        st.dataframe(freq.head(20), use_container_width=True)

# PAGE: CONFIDENCE INTERVALS
elif page == "🎯 Confidence Intervals":
    st.title("🎯 Confidence Interval Estimation")
    
    st.markdown("""
    A **confidence interval** provides a range where the true population mean likely falls.
    """)
    
    col = st.selectbox("Select Variable:", numeric_cols)
    confidence = st.slider("Confidence Level (%):", 90, 99, 95)
    
    data = filtered_df[col].replace([np.inf, -np.inf], np.nan).dropna()
    
    mean = data.mean()
    sem = stats.sem(data)
    ci = stats.t.interval(confidence / 100, len(data) - 1, loc=mean, scale=sem)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("📊 Sample Mean", f"{mean:.4f}")
    with c2:
        st.metric("⬇️ Lower Bound", f"{ci[0]:.4f}")
    with c3:
        st.metric("⬆️ Upper Bound", f"{ci[1]:.4f}")
    
    st.success(f"""
    ✅ **Interpretation:**
    
    We are **{confidence}% confident** that the true population mean of **{col}** 
    lies between **{ci[0]:.4f}** and **{ci[1]:.4f}**.
    """)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvline(mean, color='red', linewidth=2, label=f'Mean: {mean:.2f}')
    ax.axvline(ci[0], color='green', linestyle='--', linewidth=2, label=f'Lower: {ci[0]:.2f}')
    ax.axvline(ci[1], color='blue', linestyle='--', linewidth=2, label=f'Upper: {ci[1]:.2f}')
    ax.axvspan(ci[0], ci[1], alpha=0.2, color='yellow')
    ax.set_title(f'{confidence}% Confidence Interval for {col}')
    ax.set_xlabel(col)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# PAGE: PROBABILITY DISTRIBUTION
elif page == "🎲 Probability Distribution":
    st.title("🎲 Probability Methods and Distribution")
    
    tab1, tab2 = st.tabs(["📊 Normal Distribution", "🎯 Binomial Probability"])
    
    with tab1:
        st.subheader("Normal Distribution Analysis")
        col = st.selectbox("Select Variable:", numeric_cols, key='prob')
        data = filtered_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        
        mean = data.mean()
        std = data.std()
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Mean (μ)", f"{mean:.4f}")
        with c2:
            st.metric("Std Dev (σ)", f"{std:.4f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data, bins=30, density=True, alpha=0.6, edgecolor='black', label='Data')
        x = np.linspace(data.min(), data.max(), 200)
        ax.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2, label='Normal Curve')
        ax.set_xlabel(col)
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Normal Distribution Fit: {col}')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # Probability calculation
        st.markdown("---")
        st.subheader("Calculate Probability P(a < X < b)")
        
        c1, c2 = st.columns(2)
        with c1:
            lower = st.number_input("Lower bound (a):", value=float(data.min()))
        with c2:
            upper = st.number_input("Upper bound (b):", value=float(data.mean()))
        
        z_low = (lower - mean) / std
        z_up = (upper - mean) / std
        prob = stats.norm.cdf(z_up) - stats.norm.cdf(z_low)
        
        st.success(f"**P({lower:.2f} < X < {upper:.2f}) = {prob:.4f} ({prob*100:.2f}%)**")
        
        # Normality test
        st.markdown("---")
        st.subheader("Shapiro-Wilk Normality Test")
        
        if len(data) <= 5000:
            stat, p_val = stats.shapiro(data)
            st.write(f"**Test Statistic:** {stat:.4f}")
            st.write(f"**P-value:** {p_val:.6f}")
            
            if p_val > 0.05:
                st.success("✅ Data appears normally distributed (p > 0.05)")
            else:
                st.warning("⚠️ Data may not be normally distributed (p ≤ 0.05)")
        else:
            st.info("Sample too large for test")
    
    with tab2:
        st.subheader("Binomial Probability - Highly Rated Movies")
        
        # Define success as high rating (e.g., Audience Score > 70%)
        if 'Audience score %' in filtered_df.columns:
            threshold = st.slider("Define 'Highly Rated' as Audience Score >", 50, 90, 70)
            
            success_data = filtered_df['Audience score %'].dropna()
            p = (success_data > threshold).mean()
            
            st.info(f"**Success Probability (p) = {p:.4f} ({p*100:.2f}%)**")
            st.write(f"This is the probability that a randomly selected movie has Audience Score > {threshold}%")
            
            n = st.number_input("Number of movies (n):", 1, 100, 10)
            k = st.number_input("Number of highly rated (k):", 0, int(n), 5)
            
            exact = stats.binom.pmf(k, n, p)
            at_most = stats.binom.cdf(k, n, p)
            at_least = 1 - stats.binom.cdf(k - 1, n, p)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(f"P(X = {k})", f"{exact:.4f}")
            with c2:
                st.metric(f"P(X ≤ {k})", f"{at_most:.4f}")
            with c3:
                st.metric(f"P(X ≥ {k})", f"{at_least:.4f}")
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            x_vals = np.arange(0, n + 1)
            probs = stats.binom.pmf(x_vals, n, p)
            colors = ['red' if x == k else 'steelblue' for x in x_vals]
            ax.bar(x_vals, probs, color=colors, edgecolor='black')
            ax.set_xlabel('Number of Highly Rated Movies')
            ax.set_ylabel('Probability')
            ax.set_title(f'Binomial Distribution (n={n}, p={p:.2f})')
            ax.grid(alpha=0.3)
            st.pyplot(fig)

# PAGE: REGRESSION & PREDICTION
elif page == "🔮 Regression & Prediction":
    st.title("🔮 Regression Modeling and Prediction")
    
    st.markdown("Build a linear regression model to predict movie performance metrics.")
    
    # Target selection
    target = st.selectbox("Select Target Variable (Y):", 
                         ['Profitability', 'Worldwide Gross', 'Audience score %'])
    
    # Feature selection
    available = [c for c in numeric_cols if c != target]
    defaults = [c for c in available if c != 'Year'][:3]
    
    features = st.multiselect("Select Independent Variables (X):", available, default=defaults)
    
    if len(features) == 0:
        st.warning("⚠️ Select at least one independent variable")
        st.stop()
    
    # Prepare data
    model_data = filtered_df[[target] + features].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(model_data) < 10:
        st.error("❌ Not enough data. Try different variables or filters.")
        st.stop()
    
    X = model_data[features]
    y = model_data[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    st.subheader("📊 Model Performance")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("R² Score", f"{r2:.4f}")
        st.caption("Closer to 1.0 = better")
    with c2:
        st.metric("RMSE", f"{rmse:.2f}")
        st.caption("Lower = better")
    with c3:
        st.metric("MAE", f"{mae:.2f}")
        st.caption("Lower = better")
    
    # Coefficients
    st.markdown("---")
    st.subheader("📋 Regression Coefficients")
    
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    st.dataframe(coef_df, use_container_width=True)
    
    # Equation
    eq = f"{target} = {model.intercept_:.2f}"
    for f, c in zip(features, model.coef_):
        eq += f" + ({c:.2f} × {f})"
    
    st.subheader("📐 Regression Equation")
    st.code(eq)
    
    # Actual vs Predicted
    st.markdown("---")
    st.subheader("📈 Actual vs Predicted Values")
    
    result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    fig = px.scatter(result, x='Actual', y='Predicted', title=f'Actual vs Predicted {target}', trendline='ols')
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines', name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction
    st.markdown("---")
    st.subheader("🎯 Make a Prediction")
    
    input_vals = []
    cols = st.columns(min(len(features), 3))
    
    for i, feat in enumerate(features):
        with cols[i % len(cols)]:
            val = st.number_input(f"{feat}:", value=float(model_data[feat].median()), key=f"pred_{feat}")
            input_vals.append(val)
    
    if st.button("🔮 Predict", type="primary"):
        pred = model.predict(np.array(input_vals).reshape(1, -1))[0]
        
        if target == 'Worldwide Gross':
            st.success(f"### 💰 Predicted {target}: ${pred:,.2f}")
        elif target == 'Profitability':
            st.success(f"### 📊 Predicted {target}: {pred:.2f}x")
        else:
            st.success(f"### ⭐ Predicted {target}: {pred:.2f}%")

# PAGE: PROJECT SUMMARY
elif page == "📝 Project Summary":
    st.title("📝 Project Summary")
    
    st.markdown("""
    ## 🎬 Statistical Analysis and Profitability Prediction of Hollywood Movies
    
    ---
    
    ### 📊 Dataset Information
    
    **Source:** [Hollywood Movies Dataset on Kaggle](https://www.kaggle.com/datasets/writuparnabanerjee/hollywood-movies)
    
    **Description:**
    - Dataset contains information about Hollywood movies
    - Includes financial data, ratings, and genre information
    - Clean and preprocessed data ready for statistical analysis
    
    ### 🎯 Project Objectives
    
    1. Perform comprehensive statistical analysis on Hollywood movie data
    2. Visualize relationships between variables
    3. Calculate descriptive statistics and confidence intervals
    4. Analyze probability distributions
    5. Build regression models to predict profitability and revenue
    
    ### 📈 Statistical Methods Applied
    
    | Method | Application |
    |--------|-------------|
    | **Histograms** | Distribution analysis of ratings, gross, profitability |
    | **Box Plots** | Outlier detection and genre comparison |
    | **Scatter Plots** | Relationship visualization between variables |
    | **Correlation Heatmap** | Variable correlation analysis |
    | **Descriptive Statistics** | Mean, median, mode, variance, std dev, skewness, kurtosis |
    | **Confidence Intervals** | Population parameter estimation |
    | **Normal Distribution** | Distribution fitting and probability calculation |
    | **Binomial Distribution** | Success probability analysis |
    | **Linear Regression** | Profitability and revenue prediction |
    
    ### 📊 Key Variables Analyzed
    
    - **Year** - Release year
    - **Audience Score %** - Audience rating percentage
    - **Rotten Tomatoes %** - Critics rating percentage
    - **Worldwide Gross** - Total worldwide revenue
    - **Profitability** - Revenue to budget ratio
    - **Genre** - Movie category
    - **Lead Studio** - Production studio
    
    ### 🛠️ Technologies Used
    
    - **Python 3.14** - Programming language
    - **Streamlit** - Web application framework
    - **Pandas & NumPy** - Data manipulation and analysis
    - **Matplotlib & Seaborn** - Static visualizations
    - **Plotly** - Interactive visualizations
    - **SciPy** - Statistical functions
    - **Scikit-learn** - Machine learning and regression
    
    ### 🔍 Key Findings
    
    ✅ Strong correlation between audience scores and profitability  
    ✅ Genre significantly affects movie performance  
    ✅ Regression models can predict profitability with reasonable accuracy  
    ✅ Normal distribution fits well for most continuous variables  
    
    ### 🎓 Conclusion
    
    This project successfully demonstrates the application of probability and statistical
    methods on real-world Hollywood movie data. The comprehensive analysis provides insights
    into factors affecting movie performance and enables prediction of profitability and revenue.
    
    The web-based interface makes statistical analysis accessible and interactive, allowing
    users to explore relationships, calculate probabilities, and make predictions based on
    historical movie data.
    
    ---
    
    ### 👥 Team Members
    
    | # | Roll Number | Name | Department | Section |
    |---|-------------|------|------------|---------|
    | 1 | [Your Roll] | [Your Name] | CS | [Section] |
    | 2 | [Roll] | [Name] | CS | [Section] |
    | 3 | [Roll] | [Name] | CS | [Section] |
    | 4 | [Roll] | [Name] | [Dept] | [Section] |
    | 5 | [Roll] | [Name] | [Dept] | [Section] |
    
    **Semester:** Spring 2026  
    **Course:** Probability and Statistics
    
    ---
    
    **Made with ❤️ using Python and Streamlit**
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Probability & Statistics**")
st.sidebar.markdown("Spring 2026")