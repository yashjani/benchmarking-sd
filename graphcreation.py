import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from glob import glob
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Set a consistent theme for all plots
sns.set_theme(style="whitegrid")

# Path to your JSON files
file_paths = glob('/Users/Rang/Documents/benchmarking-sd/*/log/*.json')

# Print file paths to verify
print("Found files:")
for file_path in file_paths:
    print(file_path)

# Initialize an empty list to hold data from all files
all_data = []

# Load and parse each JSON file
for file_path in file_paths:
    with open(file_path) as f:
        data = json.load(f)
        for item in data['data']:
            entry = {
                'time_ms': item['time_ms'],
                'duration': item['value']['duration'] / 60,  # Convert duration from seconds to minutes
                'gpu_utilization': int(item['value']['gpu_metrics'][0]['gpu_utilization']),
                'gpu_memory_utilization': int(item['value']['gpu_metrics'][0]['gpu_memory_utilization']),
                'gpu_memory_total': int(item['value']['gpu_metrics'][0]['gpu_memory_total']),
                'gpu_memory_used': int(item['value']['gpu_metrics'][0]['gpu_memory_used']),
                'gpu_memory_free': int(item['value']['gpu_metrics'][0]['gpu_memory_free']),
                'gpu_temperature': int(item['value']['gpu_metrics'][0]['gpu_temperature']),
                'gpu_power_draw': float(item['value']['gpu_metrics'][0]['gpu_power_draw']),
                'ondemand_cost': float(item['value']['ondemand_cost']),
                'reserved_one_year_cost': float(item['value']['reserved_one_year_cost']),
                'reserved_three_year_cost': float(item['value']['reserved_three_year_cost']),
                'spot_cost': float(item['value']['spot_cost']),
                'instance_type': item['value']['instance_type'],
                'file_path': file_path  # Include file path to trace data source
            }
            all_data.append(entry)

# Create a single DataFrame
df = pd.DataFrame(all_data)

# Add image count as an index
df['image_count'] = df.groupby('instance_type').cumcount() + 1

# Create a mapping from instance type to numeric values
instance_mapping = {instance: idx for idx, instance in enumerate(df['instance_type'].unique())}
df['instance_code'] = df['instance_type'].map(instance_mapping)

# Ensure all utilization values are between 0 and 100
df['gpu_utilization'] = df['gpu_utilization'].clip(0, 100)
df['gpu_memory_utilization'] = df['gpu_memory_utilization'].clip(0, 100)

# Sum the durations for each instance type
df_sum_duration = df.groupby('instance_type')['duration'].sum().reset_index()

# Sort by duration in ascending order
df_sum_duration = df_sum_duration.sort_values(by='duration', ascending=True)

# Sum the costs for each instance type
df_sum_cost = df.groupby('instance_type')[['ondemand_cost', 'reserved_one_year_cost', 'reserved_three_year_cost', 'spot_cost']].sum().reset_index()

# Sort by on-demand cost in ascending order
df_sum_cost = df_sum_cost.sort_values(by='ondemand_cost', ascending=True)

# Create 'latency' directory if it doesn't exist
latency_dir = 'graph'
if not os.path.exists(latency_dir):
    os.makedirs(latency_dir)

# Create 'csv' directory if it doesn't exist
csv_dir = 'csv'
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Plot the latency duration graph (total duration by instance type)
plt.figure(figsize=(12, 8))
colors = cm.viridis(np.linspace(0, 1, len(df_sum_duration)))
bars = plt.bar(df_sum_duration['instance_type'], df_sum_duration['duration'], color=colors, edgecolor='black')
plt.xlabel('Instance Type')
plt.ylabel('Total Duration (minutes)')
plt.title('Total Duration by Instance Type for Creating 30 Images (in Minutes)')
plt.xticks(rotation=45)
plt.yticks(np.arange(0, df_sum_duration['duration'].max() + 1, 1))  # Adjust the interval to 1 minute
plt.tight_layout()

# Add annotations
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Save the latency duration plot as an image
plt.savefig(os.path.join(latency_dir, 'duration_by_instance_type.png'))

# Show the latency duration plot
plt.show()

# Plot the bar graph of the total costs by instance type
plt.figure(figsize=(12, 8))
bar_width = 0.15
index = np.arange(len(df_sum_cost))

# Plotting On-Demand, Reserved 1-Year, Reserved 3-Year, and Spot Costs side by side
bar1 = plt.bar(index, df_sum_cost['ondemand_cost'], bar_width, label='On-Demand Cost', color='skyblue', edgecolor='black')
bar2 = plt.bar(index + bar_width, df_sum_cost['reserved_one_year_cost'], bar_width, label='1-Year Reserved Cost', color='lightgreen', edgecolor='black')
bar3 = plt.bar(index + 2 * bar_width, df_sum_cost['reserved_three_year_cost'], bar_width, label='3-Year Reserved Cost', color='lightcoral', edgecolor='black')
bar4 = plt.bar(index + 3 * bar_width, df_sum_cost['spot_cost'], bar_width, label='Spot Cost', color='gold', edgecolor='black')

plt.xlabel('Instance Type')
plt.ylabel('Total Cost (USD)')
plt.title('Total Costs by Instance Type')
plt.xticks(index + 1.5 * bar_width, df_sum_cost['instance_type'], rotation=45)
plt.yticks(np.arange(0, df_sum_cost[['ondemand_cost', 'reserved_one_year_cost', 'reserved_three_year_cost', 'spot_cost']].max().max() + 0.1, 0.1))  # Adjust the interval to 0.1 dollars
plt.legend()

plt.tight_layout()

# Add annotations
for bars in [bar1, bar2, bar3, bar4]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, round(yval, 2), ha='center', va='bottom')

# Save the cost plot as an image
plt.savefig(os.path.join(latency_dir, 'cost_by_instance_type.png'))

# Show the cost plot
plt.show()

# 3D Surface Plot of GPU Temperature
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
x = df['image_count']
y = df['instance_code']
z = df['gpu_temperature']
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
ax.set_xlabel('Image Count')
ax.set_ylabel('Instance Type')
ax.set_zlabel('GPU Temperature (°C)')
ax.set_yticks(list(instance_mapping.values()))
ax.set_yticklabels(list(instance_mapping.keys()))
plt.title('3D Surface Plot of GPU Temperature')
plt.tight_layout()

# Save the 3D surface plot as an image
plt.savefig(os.path.join(latency_dir, 'gpu_temperature_3d_surface_plot.png'))

# Show the 3D surface plot
plt.show()

# Calculate average GPU power draw for each instance type
df_avg_power_draw = df.groupby('instance_type')['gpu_power_draw'].mean().reset_index()

# Plot the bar graph of the average GPU power draw by instance type
plt.figure(figsize=(12, 8))
colors = cm.viridis(np.linspace(0, 1, len(df_avg_power_draw)))
bars = plt.bar(df_avg_power_draw['instance_type'], df_avg_power_draw['gpu_power_draw'], color=colors, edgecolor='black')
plt.xlabel('Instance Type')
plt.ylabel('Average GPU Power Draw (Watts)')
plt.title('Average GPU Power Draw by Instance Type')
plt.xticks(rotation=45)
plt.tight_layout()

# Add annotations
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

# Save the average GPU power draw plot as an image
plt.savefig(os.path.join(latency_dir, 'avg_gpu_power_draw_by_instance_type.png'))

# Show the average GPU power draw plot
plt.show()

# Heatmap of GPU Utilization
pivot_table = df.pivot_table(values='gpu_utilization', index='instance_type', columns='image_count')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='viridis', annot=True)
plt.xlabel('Image Count')
plt.ylabel('Instance Type')
plt.title('Heatmap of GPU Utilization')
plt.tight_layout()

# Save the heatmap as an image
plt.savefig(os.path.join(latency_dir, 'gpu_utilization_heatmap.png'))

# Show the heatmap
plt.show()

# Correlation Matrix Heatmap
correlation_matrix = df[['gpu_utilization', 'gpu_memory_utilization', 'gpu_temperature', 'gpu_power_draw']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap of GPU Metrics')
plt.tight_layout()

# Save the correlation matrix heatmap as an image
plt.savefig(os.path.join(latency_dir, 'gpu_metrics_correlation_matrix.png'))

# Show the correlation matrix heatmap
plt.show()

# Write the DataFrame to a CSV file
csv_file_path = os.path.join(csv_dir, 'benchmark_data.csv')
df.to_csv(csv_file_path, index=False)

print(f"CSV file '{csv_file_path}' created successfully.")
print(f"Latency duration plot saved as '{os.path.join(latency_dir, 'duration_by_instance_type.png')}'.")
print(f"Cost plot saved as '{os.path.join(latency_dir, 'cost_by_instance_type.png')}'.")
print(f"3D surface plot of GPU temperature saved as '{os.path.join(latency_dir, 'gpu_temperature_3d_surface_plot.png')}'.")
print(f"Average GPU power draw plot saved as '{os.path.join(latency_dir, 'avg_gpu_power_draw_by_instance_type.png')}'.")
print(f"GPU utilization heatmap saved as '{os.path.join(latency_dir, 'gpu_utilization_heatmap.png')}'.")
print(f"Correlation matrix heatmap saved as '{os.path.join(latency_dir, 'gpu_metrics_correlation_matrix.png')}'.")
