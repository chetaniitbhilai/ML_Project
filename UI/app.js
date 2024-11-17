from flask import Flask, request, jsonify
import overpy
import csv

app = Flask(__name__)

def fetch(lat, lon, amenity_types, radius):
    api = overpy.Overpass()

    # Define the filename for storing all amenities in one file
    filename = 'amenities_generic.csv'

    # Create or overwrite the CSV file and write the header only once
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Amenity Type', 'Name', 'Latitude', 'Longitude'])

    # Iterate over each amenity type in the list
    for amenity_type in amenity_types:
        # Calculate the bounding box
        min_lat = lat - radius
        max_lat = lat + radius
        min_lon = lon - radius
        max_lon = lon + radius

        # Construct the Overpass query
        query = f"""
        [out:json];
        node["amenity"="{amenity_type}"]({min_lat},{min_lon},{max_lat},{max_lon});
        out body;
        """

        try:
            # Execute the query
            result = api.query(query)

            # Check if any nodes are found
            if not result.nodes:
                print(f"No {amenity_type}s found near the given coordinates.")
                continue

            # Open the CSV file in append mode to add data for each amenity type
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write each node's data
                for node in result.nodes:
                    name = node.tags.get("name", "Unknown")
                    node_lat = node.lat
                    node_lon = node.lon
                    writer.writerow([amenity_type, name, node_lat, node_lon])

            print(f"Results for {amenity_type} added to '{filename}'.")

        except overpy.exception.OverpassTooManyRequests:
            print("Too many requests to Overpass API. Please try again later.")
        except Exception as e:
            print(f"An error occurred while processing {amenity_type}: {e}")

@app.route('/api/locations', methods=['POST'])
def receive_location():
    data = request.get_json()

    lat = data['latitude']
    lon = data['longitude']
    radius_km = data['radius_km']

    amenity_types = [
        'bar', 'restaurant', 'school', 'hospital', 'park', 'atm', 'bank', 'cafe', 'gym'
    ]

    fetch(lat, lon, amenity_types, radius_km)

    return jsonify({"message": "Amenities search complete."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)