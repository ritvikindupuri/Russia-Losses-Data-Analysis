import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. Data Loading and Cleaning
# ---------------------------------------------------------
# Load the dataset
df = pd.read_csv('russia_losses.csv')

# Convert date columns to datetime objects for proper plotting
df['date_end'] = pd.to_datetime(df['date_end'])
df['date_start'] = pd.to_datetime(df['date_start'])

# Fill missing values (NaN) with 0, assuming NaN means no recorded losses
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# Calculate 'Total Equipment' losses by summing all equipment columns (excluding Personnel)
# We filter out 'Personnel' to separate human losses from machine losses
equipment_cols = [c for c in numeric_cols if c != 'Personnel']
df['Total_Equipment'] = df[equipment_cols].sum(axis=1)

# ---------------------------------------------------------
# 2. Visualizations
# ---------------------------------------------------------

# --- A. Pie Chart: Distribution of Equipment Losses ---
total_losses_per_type = df[equipment_cols].sum().sort_values(ascending=False)

# Group smaller categories into "Others" for a cleaner chart
top_n = 6
top_losses = total_losses_per_type.head(top_n)
other_losses = pd.Series([total_losses_per_type.iloc[top_n:].sum()], index=['Others'])
pie_data = pd.concat([top_losses, other_losses])

plt.figure(figsize=(10, 8))
plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Distribution of Total Equipment Losses (Top Categories)')
plt.savefig('pie_chart_equipment.png')
plt.show()

# --- B. Time Series Line Chart: Personnel vs Total Equipment ---
plt.figure(figsize=(14, 7))
plt.plot(df['date_end'], df['Personnel'], label='Personnel Losses', marker='o', linewidth=2)
plt.plot(df['date_end'], df['Total_Equipment'], label='Total Equipment Losses', marker='s', linewidth=2)
plt.title('Weekly Losses: Personnel vs Equipment')
plt.xlabel('Date (End of Week)')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('time_series_losses.png')
plt.show()

# --- C. Correlation Heatmap ---
plt.figure(figsize=(12, 10))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".1f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Loss Categories')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# ---------------------------------------------------------
# 3. Machine Learning: Anomaly Detection
# ---------------------------------------------------------

# We use Isolation Forest to detect unusual weeks based on numerical patterns
X = df[numeric_cols]

# Standardize the data (important for ML models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and fit the model
# contamination=0.1 estimates that about 10% of the data might be anomalies
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = iso_forest.fit_predict(X_scaled)

# The model outputs -1 for anomalies and 1 for normal points
anomalous_weeks = df[df['anomaly'] == -1]
normal_weeks = df[df['anomaly'] == 1]

# --- D. Visualizing Anomalies ---
plt.figure(figsize=(14, 7))
plt.plot(df['date_end'], df['Personnel'], color='gray', alpha=0.5, label='Trend')
plt.scatter(normal_weeks['date_end'], normal_weeks['Personnel'], c='blue', label='Normal Week')
plt.scatter(anomalous_weeks['date_end'], anomalous_weeks['Personnel'], c='red', s=100, label='Anomaly Detected')

# Label the anomalous dates on the plot
for idx, row in anomalous_weeks.iterrows():
    plt.annotate(row['date_end'].strftime('%Y-%m-%d'), 
                 (row['date_end'], row['Personnel']), 
                 xytext=(0,10), textcoords='offset points', ha='center', color='red')

plt.title('Anomaly Detection: Unusual Weeks Identified by Machine Learning')
plt.xlabel('Date')
plt.ylabel('Personnel Losses')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('anomaly_detection.png')
plt.show()

# Print out the results
print("Anomalous Dates Detected:")
print(anomalous_weeks[['date_end', 'Personnel', 'Total_Equipment']])
