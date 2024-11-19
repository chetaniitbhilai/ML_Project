import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import folium
import numpy as np
from scipy.spatial import distance_matrix
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
from shapely.ops import nearest_points

# Load the train data
train_data = pd.read_csv('updated_data.csv')

# Encode categorical string columns using OneHotEncoder
string_columns = train_data.select_dtypes(include=['object']).columns
if len(string_columns) > 0:
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid multicollinearity
    encoded_columns = encoder.fit_transform(train_data[string_columns])
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(string_columns))
    train_data = pd.concat([train_data.drop(columns=string_columns), encoded_df], axis=1)

# Extract necessary columns (Latitude, Longitude, Population, and any encoded features)
train_coords = train_data[['Latitude', 'Longitude', 'population'] + list(encoded_df.columns)]

# Normalize Latitude, Longitude, and Population using MinMaxScaler
scaler = MinMaxScaler()
train_coords[['Latitude', 'Longitude', 'population']] = scaler.fit_transform(
    train_coords[['Latitude', 'Longitude', 'population']]
)

# Create the graph (G) for the training dataset
G = nx.Graph()
for idx, row in train_coords.iterrows():
    G.add_node(idx, latitude=row['Latitude'], longitude=row['Longitude'], population=row['population'])

# Use Nearest Neighbors to add edges based on geographic proximity
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(train_coords[['Latitude', 'Longitude']])
distances, indices = nbrs.kneighbors(train_coords[['Latitude', 'Longitude']])
for i, neighbors in enumerate(indices):
    for j in range(1, len(neighbors)):
        # Compute the absolute difference in population
        population_diff = abs(train_coords['population'].iloc[i] + train_coords['population'].iloc[neighbors[j]])
        # Add the edge with the population difference as weight
        G.add_edge(i, neighbors[j], weight=population_diff)

# Prepare the node features and edges for PyTorch Geometric
node_features = []
for node in G.nodes(data=True):
    # Include all features: latitude, longitude, population, and encoded columns
    node_features.append([node[1]['latitude'], node[1]['longitude'], node[1]['population']])

# Adding additional encoded features to the node features
encoded_features = train_coords.drop(columns=['Latitude', 'Longitude', 'population']).values
node_features = np.concatenate([node_features, encoded_features], axis=1)

edge_index = []
edge_weight = []
for edge in G.edges(data=True):
    edge_index.append([edge[0], edge[1]])
    edge_weight.append(edge[2]['weight'])

# Convert node features, edge index, and edge weights to torch tensors
x = torch.tensor(node_features, dtype=torch.float)
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight, dtype=torch.float)

# Create the PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

# Define the GCN model with approximately 10 layers for binary classification
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(x.size(1), 16)  # Adjust input dimension dynamically
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        self.conv4 = GCNConv(64, 64)
        self.conv5 = GCNConv(64, 32)
        self.conv6 = GCNConv(32, 32)
        self.conv7 = GCNConv(32, 16)
        self.conv8 = GCNConv(16, 16)
        self.conv9 = GCNConv(16, 8)
        self.conv10 = GCNConv(8, 1)  # Single output for binary classification

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = torch.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = torch.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = torch.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
        x = torch.relu(self.conv4(x, edge_index, edge_weight=edge_weight))
        x = torch.relu(self.conv5(x, edge_index, edge_weight=edge_weight))
        x = torch.relu(self.conv6(x, edge_index, edge_weight=edge_weight))
        x = torch.relu(self.conv7(x, edge_index, edge_weight=edge_weight))
        x = torch.relu(self.conv8(x, edge_index, edge_weight=edge_weight))
        x = torch.relu(self.conv9(x, edge_index, edge_weight=edge_weight))
        x = torch.sigmoid(self.conv10(x, edge_index, edge_weight=edge_weight))  # Sigmoid activation for binary output
        return x.squeeze()  # Squeeze to make it compatible with BCE loss

# Initialize the model, BCE loss function, and optimizer
model = GCN()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Prepare binary target labels (1 for metro station, 0 for non-station)
target = torch.randint(0, 2, (data.num_nodes,), dtype=torch.float)  # Binary labels (0 or 1)

# Train the model with accuracy tracking
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data)  # Forward pass
    loss = criterion(out, target)  # BCE loss
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    with torch.no_grad():
        predicted_labels = (out > 0.5).float()  # Convert probabilities to binary (0 or 1)
        correct = (predicted_labels == target).sum().item()
        accuracy = correct / target.size(0)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy * 100:.2f}%')

# Post-process: Obtain predictions, apply threshold, and convert back to original latitude/longitude scale
model.eval()
predicted_probs = model(data).detach().numpy()
predicted_labels = (predicted_probs > 0.5).astype(int)  # Threshold at 0.5 to get binary labels

# Add the binary column for metro station prediction
train_data['Predicted_Metro_Station'] = predicted_labels

# Extract only the latitude and longitude for inverse scaling
predicted_coords = np.zeros((predicted_labels.shape[0], 3))
predicted_coords[:, :2] = train_coords[['Latitude', 'Longitude']].values
predicted_coords[:, 2] = predicted_labels  # Place binary predictions for scaling

# Inverse transform only the coordinates (Latitude and Longitude)
predicted_coords = scaler.inverse_transform(predicted_coords)
predicted_positions = predicted_coords[predicted_coords[:, 2] == 1, :2]  # Select predicted metro stations



# Check if predicted coordinates are empty
if predicted_coords.size == 0:
    print("No metro stations were predicted. Adjust the threshold or check the model output.")
else:
    # Apply KMeans clustering to group predicted metro stations
    num_clusters = min(30, len(predicted_coords))  # Ensure we don't request more clusters than data points
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(predicted_coords)
    reduced_predicted_positions = kmeans.cluster_centers_

    # Clip latitude and longitude values to valid ranges (adjust as per data ranges)
    reduced_predicted_positions[:, 0] = np.clip(reduced_predicted_positions[:, 0],
                                                train_data['Latitude'].min(), train_data['Latitude'].max())
    reduced_predicted_positions[:, 1] = np.clip(reduced_predicted_positions[:, 1],
                                                train_data['Longitude'].min(), train_data['Longitude'].max())

    # Plot the original train data and reduced predicted metro stations
    plt.figure(figsize=(8, 6))
    plt.scatter(train_data['Longitude'], train_data['Latitude'], color='blue', label='Train Stations (Original)', alpha=0.6)
    plt.scatter(reduced_predicted_positions[:, 1], reduced_predicted_positions[:, 0],
                color='red', label='Predicted Metro Stations', s=80)
    plt.title('Reduced Predicted Metro Station Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    output_path = "output_graph.png"
    plt.savefig(output_path, format="png", dpi=300)  # Save with high quality
    print(f"Graph saved as {output_path}")
    
    plt.show()


def plot_single_metro_line(predicted_positions, save_path='predicted_metro_line_map.html'):
    """
    Plot predicted metro stations on a map and connect them sequentially along actual roads.
    
    Args:
        predicted_positions: Array of predicted metro station coordinates (latitude, longitude).
        save_path: Path to save the generated map as an HTML file.
    """
    if len(predicted_positions) == 0:
        print("No predicted metro station locations available.")
        return
    
    # Extract only latitude and longitude
    predicted_positions = predicted_positions[:, :2]
    
    # Calculate the bounding box for the area
    min_lat, max_lat = predicted_positions[:, 0].min(), predicted_positions[:, 0].max()
    min_lon, max_lon = predicted_positions[:, 1].min(), predicted_positions[:, 1].max()
    
    # Add some padding to the bounding box
    padding = 0.02  # roughly 2km
    bbox = (
        min_lat - padding,
        min_lon - padding,
        max_lat + padding,
        max_lon + padding
    )
    
    # Download the street network
    G = ox.graph_from_bbox(
        bbox[0], bbox[2], bbox[1], bbox[3],
        network_type='drive',
        simplify=True
    )
    
    # Convert to projected graph for accurate distance calculations
    G_proj = ox.project_graph(G)
    
    # Find the optimal order of stations (using nearest neighbor algorithm)
    dist_matrix = distance_matrix(predicted_positions, predicted_positions)
    num_points = len(predicted_positions)
    visited = [False] * num_points
    path = [0]
    visited[0] = True
    
    for _ in range(num_points - 1):
        last_point = path[-1]
        nearest_neighbor = None
        min_distance = float('inf')
        for j in range(num_points):
            if not visited[j] and dist_matrix[last_point, j] < min_distance:
                nearest_neighbor = j
                min_distance = dist_matrix[last_point, j]
        path.append(nearest_neighbor)
        visited[nearest_neighbor] = True
    
    ordered_positions = predicted_positions[path]
    
    # Create a folium map
    center_lat = predicted_positions[:, 0].mean()
    center_lon = predicted_positions[:, 1].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Function to find nearest node in the graph
    def get_nearest_node(lat, lon):
        return ox.distance.nearest_nodes(G, lon, lat)
    
    # Plot the road-following path between consecutive stations
    for i in range(len(ordered_positions) - 1):
        start_lat, start_lon = ordered_positions[i]
        end_lat, end_lon = ordered_positions[i + 1]
        
        # Find nearest nodes in the road network
        start_node = get_nearest_node(start_lat, start_lon)
        end_node = get_nearest_node(end_lat, end_lon)
        
        try:
            # Find the shortest path between the nodes
            route = nx.shortest_path(G, start_node, end_node, weight='length')
            
            # Get the coordinates for the route
            route_coords = []
            for node in route:
                route_coords.append([G.nodes[node]['y'], G.nodes[node]['x']])
            
            # Add the route line to the map
            folium.PolyLine(
                locations=route_coords,
                color='green',
                weight=3,
                opacity=0.8,
                popup=f"Section {i+1}"
            ).add_to(m)
        except nx.NetworkXNoPath:
            print(f"No path found between stations {i+1} and {i+2}")
            # Fall back to direct line if no path is found
            folium.PolyLine(
                locations=[[start_lat, start_lon], [end_lat, end_lon]],
                color='red',
                weight=3,
                opacity=0.8,
                popup=f"Direct connection (no road path found) {i+1}"
            ).add_to(m)
    
    # Add markers for each predicted metro station
    for idx, (lat, lon) in enumerate(ordered_positions):
        folium.Marker(
            [lat, lon],
            popup=f"Predicted Metro Station {idx + 1}<br>Coordinates: ({lat:.5f}, {lon:.5f})",
            icon=folium.Icon(color='blue', icon='train')
        ).add_to(m)
    
    # Save the map
    m.save(save_path)
    print(f"Map with road-following metro line saved to {save_path}")

# Example usage:
# reduced_predicted_positions should be a numpy array of shape (n, 2) containing lat/lon coordinates
plot_single_metro_line(reduced_predicted_positions, save_path='predicted_metro_line_map.html')

