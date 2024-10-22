# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from plotnine import ggplot, aes, geom_line, theme_minimal, labs

# Set Seaborn style for consistency with Matplotlib visuals
sns.set(style='whitegrid')

# Load the dataset
data_path = "us_population_1790_to_2020.csv"  # Adjust the file path as needed
df = pd.read_csv(data_path)

# Set up Streamlit configuration
st.set_page_config(page_title="US Population Analysis Dashboard", layout="wide")

# Title and description
st.title("📊 US Population Analysis Dashboard")
st.markdown("""
Welcome to the **US Population Analysis Dashboard**! 📈 

This interactive dashboard provides insights into the historical and predicted trends of the US population. 
Navigate through different sections to explore visualizations that cover growth patterns, milestones, and future projections.
""")

# Sidebar for Navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("### Choose Analysis Section:")
year_range = st.sidebar.slider("Select Year Range:", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=(1790, 2020), step=1)
options = st.sidebar.radio("Explore Different Insights:", [
    "📊 Overview & Predictions",
    "📈 Population Growth Trends",
    "📉 Growth Rate & Moving Averages",
    "📏 Regression Analysis & Future Trends",
    "📆 Key Population Milestones",
    "🗺️ Data Visualization - Heatmap & Distributions",
    "🌐 Advanced 3D Visualizations"
])

# Filter the dataset based on the selected year range
df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

# Function to select and display the population for a specific year
def overview_predictions():
    st.header("Overview & Predictions")
    st.markdown("""
    Get a quick overview of the predicted population and compare it with actual historical data. 
    You can enter a specific year and see the predictions versus actual population records.
    """)
    year = st.number_input("Enter a Year (1790 - 2050):", min_value=1790, max_value=2050, step=1)
    if st.button("Get Predicted Population and Compare"):
        X = df[['Year']].values
        y = df['Population'].values
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        year_poly = poly.transform(np.array([[year]]))
        predicted_population = model.predict(year_poly)
        
        # Display predicted population
        st.write(f"**Predicted Population for {year}:** {int(predicted_population[0]):,} people")
        
        # Check if the actual population data is available for that year
        if year in df['Year'].values:
            actual_population = int(df[df['Year'] == year]['Population'].values[0])
            st.write(f"**Actual Population for {year}:** {actual_population:,} people")
            
            # Compare predicted and actual populations
            difference = int(predicted_population[0]) - actual_population
            st.write(f"**Difference:** {difference:,} people")
            
            # Display a comparison chart
            comparison_fig = go.Figure()
            comparison_fig.add_trace(go.Bar(
                x=['Predicted', 'Actual'],
                y=[predicted_population[0], actual_population],
                name='Population',
                marker_color=['blue', 'green']
            ))
            comparison_fig.update_layout(
                title=f"Predicted vs. Actual Population for {year}",
                xaxis_title="Type",
                yaxis_title="Population",
                template='plotly_dark'
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
        else:
            st.write("**Actual population data for this year is not available.**")

# Function for Population Growth Trends
def population_growth_trends():
    st.header("Population Growth Trends Over Time")
    st.markdown("""
    Explore the historical trends of US population growth from 1790 to 2020. 
    This section provides line graphs to visualize how the population has evolved over time.
    """)
    
    fig = px.line(df_filtered, x='Year', y='Population', title='US Population Over Time (Interactive)',
                  labels={'Population': 'Population (in millions)', 'Year': 'Year'},
                  line_shape='spline', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x='Year', y='Population', color='blue')
    plt.title('US Population Over Time')
    plt.xlabel('Year')
    plt.ylabel('Population (in millions)')
    plt.grid(True)
    st.pyplot(plt)

# Function for Regression Analysis
def regression_analysis():
    st.header("Regression Analysis & Future Trends")
    st.markdown("""
    This section uses regression models to predict future population trends up to 2050. 
    Adjust the polynomial degree for different regression curves.
    """)
    degree = st.slider("Select Polynomial Degree:", min_value=1, max_value=10, value=2)
    
    X = df[['Year']].values
    y = df['Population'].values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    future_years = np.arange(1790, 2050).reshape(-1, 1)
    future_years_poly = poly.transform(future_years)
    predicted_population = model.predict(future_years_poly)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Actual Data', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_years.flatten(), y=predicted_population, mode='lines', name='Predicted Trend',
                             line=dict(color='red', dash='dash', width=3)))
    fig.update_layout(title=f'Population Predictions (1790 - 2050)',
                      xaxis_title='Year', yaxis_title='Population (in millions)', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    predicted_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted_Population': predicted_population})
    st.dataframe(predicted_df)

# Function for Key Population Milestones
def population_milestones():
    st.header("Key Population Milestones")
    st.markdown("""
    Explore significant milestones in population growth over time. 
    See how the population reached different levels and the periods of rapid growth.
    """)
    milestones = df_filtered[(df_filtered['Population'] > 100000000) & (df_filtered['Population'] < 350000000)]
    
    fig = px.scatter(milestones, x='Year', y='Population', title='Population Milestones Over Time (Interactive)',
                     labels={'Population': 'Population (in millions)', 'Year': 'Year'},
                     size='Population', color='Population', color_continuous_scale='Viridis', template='plotly_dark')
    fig.add_trace(go.Scatter(x=df_filtered['Year'], y=df_filtered['Population'], mode='lines', name='Population Trend',
                             line=dict(color='blue', width=2)))
    st.plotly_chart(fig, use_container_width=True)

# Function for Growth Rate and Moving Averages
def growth_rate_moving_averages():
    st.header("Growth Rate & Moving Averages")
    st.markdown("""
    Examine the annual growth rate of the US population and the 10-year moving averages to identify steady growth patterns.
    """)
    df_filtered['Annual_Growth_Rate'] = df_filtered['Population'].pct_change() * 100
    df_filtered['Moving_Average'] = df_filtered['Population'].rolling(window=10).mean()
    
    # Annual Growth Rate Bar Chart
    fig = px.bar(df_filtered, x='Year', y='Annual_Growth_Rate',
                 title='Annual Growth Rate of US Population',
                 labels={'Annual_Growth_Rate': 'Growth Rate (%)', 'Year': 'Year'},
                 color='Annual_Growth_Rate', color_continuous_scale='Bluered', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    # Moving Average Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered['Year'], y=df_filtered['Population'], mode='lines', name='Actual Population', 
                             line=dict(color='orange', width=3)))
    fig.add_trace(go.Scatter(x=df_filtered['Year'], y=df_filtered['Moving_Average'], mode='lines', name='10-Year Moving Average',
                             line=dict(color='green', dash='dash', width=3)))
    fig.update_layout(title='Population & 10-Year Moving Average',
                      xaxis_title='Year', yaxis_title='Population (in millions)',
                      template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# Function for Heatmap (Correlation)
def heatmap():
    st.header("Heatmap of Population Data")
    st.markdown("""
    This heatmap shows the correlation between different variables in the dataset, helping to identify patterns and relationships.
    """)
    corr = df_filtered[['Year', 'Population']].corr()
    
    # Seaborn Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap")
    st.pyplot(plt)

# Function for Box Plot (Distribution of Population by Year)
def box_plot():
    st.header("Box Plot of Population by Year")
    st.markdown("""
    This box plot illustrates the distribution of population by year, showing variations, outliers, and patterns over time.
    """)
    
    # Plotly Box Plot
    fig = px.box(df_filtered, x='Year', y='Population', points="all",
                 title="Box Plot of Population by Year (Interactive)",
                 labels={'Population': 'Population (in millions)', 'Year': 'Year'},
                 color='Year', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# Function for Violin Plot (Distribution of Population by Year)
def violin_plot():
    st.header("Violin Plot of Population by Year")
    st.markdown("""
    The violin plot provides a visual summary of the population distribution by year, highlighting density and range.
    """)
    fig = px.violin(df_filtered, x='Year', y='Population', box=True, points='all',
                    title="Violin Plot of Population by Year (Interactive)",
                    labels={'Population': 'Population (in millions)', 'Year': 'Year'},
                    color='Year', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# Function for Scatter Plot (Year vs. Population with Trend Line)
def scatter_plot():
    st.header("Scatter Plot of Year vs. Population")
    st.markdown("""
    This scatter plot visualizes the relationship between the year and the population, with a trend line to show overall growth.
    """)
    fig = px.scatter(df_filtered, x='Year', y='Population', trendline='ols', 
                     title="Scatter Plot of Year vs. Population (Interactive)",
                     labels={'Population': 'Population (in millions)', 'Year': 'Year'},
                                          template='plotly_dark', color='Population', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

# Function for Advanced 3D Visualizations
# Function for Advanced 3D Visualizations
def advanced_3d_visualizations():
    st.header("Advanced 3D Visualizations")
    st.markdown("""
    Experience a more detailed analysis of the population data using 3D scatter, surface plots, subplots, and animated visuals. 
    These visualizations help explore data trends and densities over time.
    """)

    # Ensure Annual Growth Rate is calculated
    df_filtered['Annual_Growth_Rate'] = df_filtered['Population'].pct_change() * 100

    # 3D Scatter Plot
    fig_3d_scatter = px.scatter_3d(
        df_filtered, x='Year', y='Population', z=df_filtered['Population'].diff().fillna(0),
        color='Population', title="3D Scatter Plot of Population",
        labels={'Population': 'Population (in millions)', 'Year': 'Year', 'z': 'Change in Population'},
        template='plotly_dark'
    )
    st.plotly_chart(fig_3d_scatter, use_container_width=True)

    # 3D Surface Plot
    years = np.linspace(df_filtered['Year'].min(), df_filtered['Year'].max(), len(df_filtered))
    populations = df_filtered['Population'].to_numpy()
    x, y = np.meshgrid(years, populations)
    z = np.outer(populations, np.ones(len(years)))

    fig_3d_surface = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig_3d_surface.update_layout(
        title='3D Surface Plot of Population Over Years',
        scene=dict(xaxis_title='Year', yaxis_title='Population', zaxis_title='Density'),
        template='plotly_dark'
    )
    st.plotly_chart(fig_3d_surface, use_container_width=True)

    # Parallel Coordinates Plot
    st.subheader("Parallel Coordinates Plot of Population")
    st.markdown("""
    A parallel plot helps in understanding how various dimensions of data relate to each other. Here, we are visualizing
    how the year, population, and growth rate interact.
    """)
    df_filtered['Decade'] = (df_filtered['Year'] // 10) * 10  # Add decade for color grouping
    fig_parallel = px.parallel_coordinates(
        df_filtered,
        dimensions=['Year', 'Population', 'Annual_Growth_Rate'],
        color='Decade',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig_parallel, use_container_width=True)

    # 3D Subplots (Scatter + Surface)
    st.subheader("Combined 3D Scatter and Surface Plot")
    st.markdown("""
    A composite view of population changes over time using both 3D scatter and surface plots. 
    This view provides a multi-dimensional perspective of the dataset.
    """)
    fig_3d_combined = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'surface'}]],
        subplot_titles=("3D Scatter Plot", "3D Surface Plot")
    )

    fig_3d_combined.add_trace(
        go.Scatter3d(x=df_filtered['Year'], y=df_filtered['Population'], z=df_filtered['Population'].diff().fillna(0),
                     mode='markers', marker=dict(size=4, color=df_filtered['Population'], colorscale='Viridis')),
        row=1, col=1
    )

    fig_3d_combined.add_trace(
        go.Surface(z=z, x=x, y=y, colorscale='Viridis'),
        row=1, col=2
    )

    fig_3d_combined.update_layout(
        title='Combined 3D Subplots of Population Trends',
        template='plotly_dark'
    )
    st.plotly_chart(fig_3d_combined, use_container_width=True)


# Function for Animated Population Plot
def animated_population_plot():
    st.header("Animated Population Growth Over Time")
    st.markdown("""
    This animated plot shows how the US population has changed over time. 
    Watch the dots grow as the years progress, reflecting changes in population dynamics.
    """)
    fig_animated = px.scatter(
        df_filtered, x='Year', y='Population', animation_frame='Year',
        size='Population', color='Population',
        range_y=[df['Population'].min(), df['Population'].max()],
        title='Animated Population Growth Over Time',
        labels={'Population': 'Population (in millions)', 'Year': 'Year'},
        template='plotly_dark'
    )
    st.plotly_chart(fig_animated, use_container_width=True)

# Function for ggplot2-style Visualization
def ggplot2_style():
    st.header("GGPlot2 Style Population Plot")
    st.markdown("""
    A classic ggplot2-style plot using the `plotnine` library in Python. 
    This mimics the aesthetics of the popular R library for elegant visualizations.
    """)
    plot = (
        ggplot(df_filtered) +
        aes(x='Year', y='Population') +
        geom_line(color='blue', size=1.2) +
        labs(title="US Population Growth (ggplot2 Style)", x="Year", y="Population (in millions)") +
        theme_minimal()
    )
    st.pyplot(plot.draw())

# Render appropriate section based on sidebar selection
if options == "📊 Overview & Predictions":
    overview_predictions()
elif options == "📈 Population Growth Trends":
    population_growth_trends()
elif options == "📉 Growth Rate & Moving Averages":
    growth_rate_moving_averages()
elif options == "📏 Regression Analysis & Future Trends":
    regression_analysis()
elif options == "📆 Key Population Milestones":
    population_milestones()
elif options == "🗺️ Data Visualization - Heatmap & Distributions":
    heatmap()
    box_plot()
    violin_plot()
    scatter_plot()
elif options == "🌐 Advanced 3D Visualizations":
    advanced_3d_visualizations()
    animated_population_plot()
    ggplot2_style()

# Footer with additional information or links
st.sidebar.markdown("""
---
Developed by [Bhavya](https://github.com/bhavya1005)
""")

                     
