# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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
Welcome to the US Population Analysis Dashboard! 📈 

This interactive dashboard provides insights into the historical and predicted trends of the US population. 
Navigate through different sections to explore various visualizations that cover growth patterns, milestones, and future projections.
""")

# Sidebar for Navigation
st.sidebar.title("Navigation")
year_range = st.sidebar.slider("Select Year Range:", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=(1790, 2020), step=1)
options = st.sidebar.radio("Select Analysis:", [
    "Home",
    "Population Over Time",
    "Moving Average",
    "Annual Growth Rate",
    "Regression Prediction",
    "Population Milestones",
    "Heatmap",
    "Box Plot",
    "Violin Plot",
    "Scatter Plot",
    "3D Scatter Plot",
    "3D Surface Plot"
])

# Filter the dataset based on the selected year range
df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

# Function to select and display the population for a specific year
def home():
    st.header("Predict and Compare Population for a Specific Year")
    year = st.number_input("Enter a Year (1790 - 2050):", min_value=1790, max_value=2050, step=1)
    if st.button("Get Predicted Population and Compare"):
        # Train a polynomial regression model with degree 2 as default
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
        actual_population_row = df[df['Year'] == year]
        if not actual_population_row.empty:
            actual_population = int(actual_population_row['Population'].values[0])
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

# Function for Population Over Time
def population_over_time():
    st.header("US Population Over Time")
    st.markdown("""
    This plot shows the historical trend of the US population from 1790 to 2020. 
    The line graph below allows you to observe how the population has grown over time.
    """)
    
    # Plotly Interactive Line Plot
    fig = px.line(df_filtered, x='Year', y='Population', title='US Population Over Time (Interactive)',
                  labels={'Population': 'Population (in millions)', 'Year': 'Year'},
                  line_shape='spline', template='plotly_dark')
    fig.update_traces(line=dict(color='blue', width=3))
    st.plotly_chart(fig, use_container_width=True)

    # Static Line Plot with Matplotlib
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x='Year', y='Population', color='blue')
    plt.title('US Population Over Time')
    plt.xlabel('Year')
    plt.ylabel('Population (in millions)')
    plt.grid(True)
    st.pyplot(plt)

# Function for Moving Average Analysis
def moving_average():
    st.header("10-Year Moving Average of Population")
    st.markdown("""
    The moving average plot smooths out short-term fluctuations to reveal longer-term trends in population growth. 
    By using a 10-year window, this graph helps to identify steady growth patterns over time.
    """)
    df_filtered['Moving_Average'] = df_filtered['Population'].rolling(window=10).mean()
    
    # Plotly Interactive Line Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered['Year'], y=df_filtered['Population'], mode='lines', name='Actual Population', 
                             line=dict(color='orange', width=3)))
    fig.add_trace(go.Scatter(x=df_filtered['Year'], y=df_filtered['Moving_Average'], mode='lines', name='10-Year Moving Average',
                             line=dict(color='green', dash='dash', width=3)))
    fig.update_layout(title='Interactive Population with 10-Year Moving Average',
                      xaxis_title='Year', yaxis_title='Population (in millions)',
                      template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# Function to calculate and display annual growth rate
def annual_growth_rate():
    st.header("Annual Growth Rate of US Population")
    st.markdown("""
    The annual growth rate provides a year-by-year comparison of how the population changes. 
    Positive values indicate growth, while negative values highlight periods where the population declined.
    """)
    df_filtered['Annual_Growth_Rate'] = df_filtered['Population'].pct_change() * 100
    
    # Plotly Interactive Bar Chart with bins
    fig = px.bar(df_filtered, x='Year', y='Annual_Growth_Rate', title='Annual Growth Rate of US Population (Interactive)',
                 labels={'Annual_Growth_Rate': 'Growth Rate (%)', 'Year': 'Year'},
                 color='Annual_Growth_Rate', color_continuous_scale='Bluered', template='plotly_dark')
    fig.update_traces(marker=dict(line=dict(width=1, color='black')))
    st.plotly_chart(fig, use_container_width=True)

# Function for Regression Prediction with Degree Selection
def regression_prediction():
    st.header("Regression Analysis: Predicting US Population (1710-2024)")
    degree = st.slider("Select Polynomial Degree:", min_value=1, max_value=10, value=2)
    
    X = df[['Year']].values
    y = df['Population'].values

    # Train a polynomial regression model
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    # Predict for future years
    future_years = np.arange(1710, 2025).reshape(-1, 1)
    future_years_poly = poly.transform(future_years)
    predicted_population = model.predict(future_years_poly)

    # Plotly Interactive Actual vs. Predicted Line Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Actual Data', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_years.flatten(), y=predicted_population, mode='lines', name='Predicted Trend',
                             line=dict(color='red', dash='dash', width=3)))
    fig.update_layout(title=f'Interactive Actual vs. Predicted Population (1710 - 2024) - Polynomial Degree {degree}',
                      xaxis_title='Year', yaxis_title='Population (in millions)', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # Show predicted data in a table
    predicted_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted_Population': predicted_population})
    st.dataframe(predicted_df)

# Function to identify and display population milestones
def population_milestones():
    st.header("Population Milestones Over Time")
    st.markdown("""
    This section highlights significant population milestones, such as the crossing of 100 million or 200 million people. 
    Milestones can help to understand the key periods of rapid growth.
    """)
    milestones = df_filtered[(df_filtered['Population'] > 100000000) & (df_filtered['Population'] < 350000000)]
    
    # Plotly Interactive Scatter Plot
    fig = px.scatter(milestones, x='Year', y='Population', title='Population Milestones Over Time (Interactive)',              
                     labels={'Population': 'Population (in millions)', 'Year': 'Year'},
                     size='Population', color='Population', color_continuous_scale='Viridis', template='plotly_dark')
    fig.add_trace(go.Scatter(x=df_filtered['Year'], y=df_filtered['Population'], mode='lines', name='Population Trend',
                             line=dict(color='blue', width=2)))
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

def scatter_3d_plot():
    st.header("3D Scatter Plot of Population Trends")
    st.markdown("""
    This 3D scatter plot provides a multi-dimensional view of population trends, helping to visualize how data varies across three axes.
    """)
    fig = px.scatter_3d(df_filtered, x='Year', y='Population', z=df_filtered['Population'].diff().fillna(0),
                        color='Population', title="3D Scatter Plot of Year vs. Population",
                        labels={'Population': 'Population (in millions)', 'Year': 'Year', 'z': 'Change in Population'},
                        template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# Function for 3D Surface Plot
def surface_3d_plot():
    st.header("3D Surface Plot of Population Trends")
    st.markdown("""
    This 3D surface plot provides an in-depth view of how population data changes over the years, showing a continuous landscape of growth.
    """)
    # Create grid for 3D surface
    years = np.linspace(df_filtered['Year'].min(), df_filtered['Year'].max(), len(df_filtered))
    populations = df_filtered['Population'].to_numpy()
    x, y = np.meshgrid(years, populations)
    z = np.outer(populations, np.ones(len(years)))

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(title='3D Surface Plot of Population Over Years',
                      scene=dict(xaxis_title='Year', yaxis_title='Population', zaxis_title='Density'),
                      template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# Render appropriate section based on sidebar selection
if options == "Home":
    home()
elif options == "Population Over Time":
    population_over_time()
elif options == "Moving Average":
    moving_average()
elif options == "Annual Growth Rate":
    annual_growth_rate()
elif options == "Regression Prediction":
    regression_prediction()
elif options == "Population Milestones":
    population_milestones()
elif options == "Heatmap":
    heatmap()
elif options == "Box Plot":
    box_plot()
elif options == "Violin Plot":
    violin_plot()
elif options == "Scatter Plot":
    scatter_plot()
elif options == "3D Scatter Plot":
    scatter_3d_plot()
elif options == "3D Surface Plot":
    surface_3d_plot()

# Footer with additional information or links
st.sidebar.markdown("""
---
Developed by [Bhavya](https://github.com/bhavya1005)
""")
