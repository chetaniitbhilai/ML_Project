
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'df' and has 'latitude' and 'longitude' columns

# Choose the number of clusters (you can experiment with different values)
n_clusters = 50

# Create a KMeans model
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# Fit the model to your data (latitude and longitude columns)
kmeans.fit(df[['latitude', 'longitude']])

# Get the cluster labels for each data point
df['cluster'] = kmeans.labels_

# Get the cluster centers
cluster_centers = kmeans.cluster_centers_

# Plot the data points with different colors for each cluster
plt.scatter(df['longitude'], df['latitude'], c=df['cluster'], cmap='viridis')

# Plot the cluster centers as crosses
plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], marker='x', s=200, color='red')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('KMeans Clustering of Amenities')
plt.show()

amenities = [
    "bar",
    "cafe",
    "fast_food",
    "food_court",
    "ice_cream",
    "pub",
    "restaurant",
    "college",
    "kindergarten",
    "language_school",
    "library",
    "training",
    "music_school",
    "school",
    "bicycle_parking",
    "fuel",
    "parking",
    "taxi",
    "atm",
    "bank",
    "money_transfer",
    "clinic",
    "dentist",
    "doctors",
    "hospital",
    "nursing_home",
    "pharmacy",
    "veterinary",
    "arts_centre",
    "cinema",
    "community_centre",
    "conference_centre",
    "fountain",
    "nightclub",
    "studio",
    "theatre",
    "courthouse",
    "fire_station",
    "police",
    "post_box",
    "post_office",
    "drinking_water",
    "shelter",
    "telephone",
    "toilets",
    "crematorium",
    "grave_yard",
    "internet_cafe",
    "marketplace",
    "place_of_worship",
    "public_bath",
    "public_building"
]
node_features = []
for cluster_id in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster_id]

    # Count the number of each amenity type in the cluster
    amenity_counts = cluster_data.groupby('Amenity Type')['Amenity Type'].count().to_dict()

    # Get the centroid location for the cluster
    centroid_lat = cluster_centers[cluster_id][0]
    centroid_lon = cluster_centers[cluster_id][1]


    # Create a feature vector for the cluster
    feature_vector = [
        cluster_id,  # Cluster ID
        len(cluster_data),  # Number of amenities in the cluster
        centroid_lat,  # Centroid latitude
        centroid_lon  # Centroid longitude
    ]

    # Add counts of different amenity types as features
    for amenity_type in amenities:
        if amenity_type in amenity_counts:
            feature_vector.append(amenity_counts[amenity_type])
        else:
            feature_vector.append(0)

    node_features.append(feature_vector)


# Create column names for the DataFrame
column_names = ['Cluster ID', 'Num Amenities', 'Centroid Latitude', 'Centroid Longitude']
column_names.extend(amenities)

# Convert the list of feature vectors to a DataFrame for easier manipulation
node_feature_df = pd.DataFrame(node_features, columns=column_names)

node_feature_df
