# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Set Seaborn style
sns.set(style='whitegrid')

# Load the dataset
data_path = "us_population_1790_to_2020.csv"  # Adjust the file path as needed
df = pd.read_csv(data_path)

# Display basic information about the data
print(df.head())
print(df.info())
print(df.describe())

# Handle any missing data (if needed)
df.dropna(inplace=True)

# 1. Matplotlib/Seaborn: Population Over Time
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Year', y='Population')
plt.title('US Population Over Time')
plt.xlabel('Year')
plt.ylabel('Population (in millions)')
plt.grid(True)
plt.savefig('population_over_time.png')
plt.show()

# 2. Plotly: Interactive Population Over Time
fig = px.line(df, x='Year', y='Population', title='US Population Over Time (Interactive)',
              labels={'Population': 'Population (in millions)', 'Year': 'Year'})
fig.update_traces(line=dict(color='blue'))
fig.show()

# 3. Matplotlib/Seaborn: Moving Average (10-year window)
df['Moving_Average'] = df['Population'].rolling(window=10).mean()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Year', y='Population', label='Actual Population')
sns.lineplot(data=df, x='Year', y='Moving_Average', label='10-Year Moving Average', linestyle='--')
plt.title('Population with 10-Year Moving Average')
plt.xlabel('Year')
plt.ylabel('Population (in millions)')
plt.legend()
plt.grid(True)
plt.savefig('moving_average.png')
plt.show()

# 4. Plotly: Interactive Moving Average
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Year'], y=df['Population'], mode='lines', name='Actual Population'))
fig.add_trace(go.Scatter(x=df['Year'], y=df['Moving_Average'], mode='lines', name='10-Year Moving Average',
                         line=dict(dash='dash')))
fig.update_layout(title='Interactive Population with 10-Year Moving Average',
                  xaxis_title='Year', yaxis_title='Population (in millions)')
fig.show()

# 5. Matplotlib/Seaborn: Annual Growth Rate
df['Annual_Growth_Rate'] = df['Population'].pct_change() * 100

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Year', y='Annual_Growth_Rate', color='skyblue')
plt.title('Annual Growth Rate of US Population')
plt.xlabel('Year')
plt.ylabel('Growth Rate (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig('annual_growth_rate.png')
plt.show()

# 6. Plotly: Interactive Annual Growth Rate
fig = px.bar(df, x='Year', y='Annual_Growth_Rate', title='Annual Growth Rate of US Population (Interactive)',
             labels={'Annual_Growth_Rate': 'Growth Rate (%)', 'Year': 'Year'})
fig.update_traces(marker_color='skyblue')
fig.show()

# 7. Regression Analysis: Predict Future Trends
X = df[['Year']].values
y = df['Population'].values

# Initialize and train the regression model
model = LinearRegression()
model.fit(X, y)

# Predict from 1710 to 2024
future_years = np.arange(1710, 2025).reshape(-1, 1)
predicted_population = model.predict(future_years)

# 8. Matplotlib/Seaborn: Actual vs. Predicted Population
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(future_years, predicted_population, color='red', linestyle='--', label='Predicted Trend')
plt.title('Actual vs. Predicted Population (1710 - 2024)')
plt.xlabel('Year')
plt.ylabel('Population (in millions)')
plt.legend()
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.show()

# 9. Plotly: Interactive Actual vs. Predicted Population
fig = go.Figure()
fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Actual Data', marker=dict(color='blue')))
fig.add_trace(go.Scatter(x=future_years.flatten(), y=predicted_population, mode='lines', name='Predicted Trend',
                         line=dict(color='red', dash='dash')))
fig.update_layout(title='Interactive Actual vs. Predicted Population (1710 - 2024)',
                  xaxis_title='Year', yaxis_title='Population (in millions)')
fig.show()

# 10. Matplotlib/Seaborn: Population Milestones
milestones = df[(df['Population'] > 100000000) & (df['Population'] < 350000000)]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=milestones, x='Year', y='Population', s=100, color='red', label='Milestones')
sns.lineplot(data=df, x='Year', y='Population')
plt.title('Population Milestones Over Time')
plt.xlabel('Year')
plt.ylabel('Population (in millions)')
plt.legend()
plt.grid(True)
plt.savefig('population_milestones.png')
plt.show()

# 11. Plotly: Interactive Population Milestones
fig = px.scatter(milestones, x='Year', y='Population', title='Population Milestones Over Time (Interactive)',
                 labels={'Population': 'Population (in millions)', 'Year': 'Year'})
fig.add_trace(go.Scatter(x=df['Year'], y=df['Population'], mode='lines', name='Population Trend', line=dict(color='blue')))
fig.show()

# Save Predictions to CSV
predicted_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted_Population': predicted_population})
predicted_df.to_csv("predicted_us_population_1710_to_2024.csv", index=False)
print(predicted_df)
