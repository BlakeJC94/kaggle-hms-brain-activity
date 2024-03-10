import numpy as np
import plotly.graph_objects as go

# Your histogram counts data
histogram_counts_data = [10, 20, 30, 40, 30, 20, 10]
bin_edges = [0, 1, 2, 3, 4, 5, 6, 7]  # Example bin edges

# Convert histogram counts to data points
data_points = []
for count, edge in zip(histogram_counts_data, bin_edges[:-1]):
    data_points.extend([edge] * count)

# Calculate the center of the violin plot
violin_center = (max(data_points) + min(data_points)) / 2

# Create a violin plot trace
fig = go.Figure()

# Add violin plot trace
fig.add_trace(go.Violin(y=data_points, name='Violin Plot'))

# Calculate normalized heights for the histogram bars
normalized_counts = [count / sum(histogram_counts_data) for count in histogram_counts_data]

# Add horizontal bar chart trace for histogram
fig.add_trace(go.Bar(y=bin_edges[:-1], x=normalized_counts, orientation='h', name='Normalized Histogram'))

# Update layout
fig.update_layout(title="Violin Plot with Overlayed Normalized Histogram",
                  yaxis_title="Data Points",
                  xaxis_title="")

# Show the plot
fig.show()

