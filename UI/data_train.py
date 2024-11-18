import overpy
import csv
import sys


sys.stdout.reconfigure(encoding='utf-8') 

def fetch(lat, lon, amenity_types, radius):
    api = overpy.Overpass()

    # Define the filename for storing all amenities in one file
    filename = 'amenities_generic.csv'

    # Create or overwrite the CSV file and write the header only once
    with open(filename, 'w', newline='',encoding='utf-8') as csvfile:
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

if __name__ == "__main__":
    print("Datatrain is running")
    try:
        # lat = float(input("Enter the latitude (e.g., 28.6139 for Delhi): "))
        # lon = float(input("Enter the longitude (e.g., 77.2090 for Delhi): "))

        # List of amenities
        amenity_types = [
            'bar', 'biergarten', 'cafe', 'fast_food', 'food_court', 'ice_cream', 'pub', 'restaurant',
            'college', 'dancing_school', 'driving_school', 'first_aid_school', 'kindergarten', 'language_school',
            'library', 'surf_school', 'toy_library', 'research_institute', 'training', 'music_school', 'school',
            'university', 'bicycle_parking', 'bicycle_repair_station', 'bicycle_wash', 'boat_rental', 'vehicle_inspection',
            'charging_station', 'driver_training', 'ferry_terminal', 'fuel', 'grit_bin', 'motorcycle_parking', 'parking',
            'parking_entrance', 'parking_space', 'taxi', 'atm', 'payment_terminal', 'bank', 'money_transfer', 'payment_centre',
            'clinic', 'dentist', 'doctors', 'hospital', 'nursing_home', 'pharmacy', 'veterinary', 'arts_centre', 'brothel',
            'casino', 'cinema', 'community_centre', "conference_centre", "events_venue", "exhibition_centre", "fountain",
            "gambling", "love_hotel", "music_venue", "nightclub", "planetarium", "public_bookcase", "social_centre",
            "stage", "stripclub", "studio", "swingerclub", "theatre", "courthouse", "fire_station", "police", "post_box",
            "post_depot", "post_office", "prison", "bbq", "bench", "dog_toilet", "dressing_room", "drinking_water",
            "give_box", "mailroom", "parcel_locker", "shelter", "shower", "telephone", "toilets", "water_point",
            "watering_place", "sanitary_dump_station", "recycling", "waste_basket", "waste_disposal", "waste_transfer_station",
            "animal_boarding", "animal_breeding", "animal_shelter", "animal_training", "baking_oven", "clock", "crematorium",
            "dive_centre", "funeral_hall", "grave_yard", "hunting_stand", "internet_cafe", "kitchen", "kneipp_water_cure",
            "lounger", "marketplace", "monastery", "mortuary", "photo_booth", "place_of_mourning", "place_of_worship",
            "public_bath", "public_building", "refugee_site", "vending_machine"
        ]

        # radius = float(input("Enter the search radius in degrees (scale: 0.1 for approximately 10 km): "))
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
        radius = float(sys.argv[3])
        fetch(lat, lon, amenity_types, radius)

    except ValueError:
        print("Invalid input! Please make sure to enter valid numbers for latitude, longitude, and radius.")

import csv

def filter_unknown_names(filename):
    # Create a temporary list to store rows that have a valid name
    filtered_rows = []

    # Read data from the existing CSV file
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the header
        for row in reader:
            # Check if the 'Name' field is not "Unknown"
            if row[1] != "Unknown":
                filtered_rows.append(row)

    # Write the filtered data back to the same CSV file
    newfilename=filename.split('.')[0]+'_filtered.csv'
    with open(newfilename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Write the header again
        writer.writerows(filtered_rows)  # Write the filtered rows

    print(f"Entries with 'Unknown' names have been removed from '{filename}'.")

# Call the function with the path to your CSV file
filter_unknown_names('amenities_generic.csv')


import os
import requests
import pandas as pd

def get_location_data(lat, lon, api_key):
    """
    Fetches the location data using the OpenCage API and returns road, suburb, and state district.
    """
    url = f"https://api.opencagedata.com/geocode/v1/json?q={lat}+{lon}&key={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        location_data = response.json()

        if location_data and location_data.get("results"):
            road = location_data["results"][0]["components"].get("road", "")
            suburb = location_data["results"][0]["components"].get("suburb", "")
            state_district = location_data["results"][0]["components"].get("state_district", "")
            return road, suburb, state_district
        else:
            return "", "", ""
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return "", "", ""
    except Exception as err:
        print(f"An error occurred: {err}")
        return "", "", ""

# List of API keys to use
api_keys = [
    "209fd3b3c460499e91f4fc86e110513b",  # Replace with your actual keys
    "b0c319e774e24be18d742768cdb5877f",
    "d3685354cdb0409d81604db1cd240cdc",
    # Add more keys if available
]

# Input file path
input_file = 'amenities_generic_filtered.csv'
output_file = 'updated_amenities_generic.csv'
api_key_index = 0
requests_count = 0
max_requests_per_key = 2400

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: The input file '{input_file}' does not exist.")
else:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file,encoding='latin1')

    # Create new columns for road, suburb, and state_district
    df['road'] = ''
    df['suburb'] = ''
    df['state_district'] = ''

    # Iterate through each row in the DataFrame to fetch the location data
    for index, row in df.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']

        # Get the current API key
        current_api_key = api_keys[api_key_index]

        # Get location details from the API
        road, suburb, state_district = get_location_data(lat, lon, current_api_key)

        # Update the DataFrame with the fetched data
        df.at[index, 'road'] = road
        df.at[index, 'suburb'] = suburb
        df.at[index, 'state_district'] = state_district

        # Update the request count
        requests_count += 1

        # Check if we need to switch the API key
        if requests_count >= max_requests_per_key:
            requests_count = 0  # Reset the count
            api_key_index = (api_key_index + 1) % len(api_keys)  # Move to the next API key
            print(f"Switching to API key: {api_keys[api_key_index]}")

        print(f"Processed {index + 1}/{len(df)}: Road={road}, Suburb={suburb}, State District={state_district}")

    # Write the updated data back to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Updated data saved to {output_file}")


import pandas as pd

# Open the CSV file
input_file = 'updated_amenities_generic.csv'
df = pd.read_csv(input_file)

# Create dictionaries counting the occurrences of each unique value in 'road', 'suburb', and 'state_district'
road_counts = df['road'].value_counts().to_dict()
suburb_counts = df['suburb'].value_counts().to_dict()
district_counts = df['state_district'].value_counts().to_dict()

# Define a function to map values with a default of 0 for unknowns
def map_with_default(value, count_dict):
    return count_dict.get(value, 0)

# Add new columns 'road_count', 'suburb_count', and 'district_count' to the DataFrame
df['road_count'] = df['road'].map(lambda x: map_with_default(x, road_counts))
df['suburb_count'] = df['suburb'].map(lambda x: map_with_default(x, suburb_counts))
df['district_count'] = df['state_district'].map(lambda x: map_with_default(x, district_counts))

# Save the updated DataFrame to a new CSV file
output_file = 'final_combined_amenities.csv'
df.to_csv(output_file, index=False)

print(f"Updated data with road, suburb, and district counts saved to {output_file}")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset (replace 'your_data.csv' with your actual file path)
data = pd.read_csv('final_combined_amenities.csv')

# Display the first few rows to verify the data structure
print(data.head())

# Scale the 'suburb' column to a range of 0 to 10
scaler = MinMaxScaler(feature_range=(0, 10))
data['scaled_suburb'] = scaler.fit_transform(data[['suburb_count']])

# Calculate the 'population' using the given formula
data['population'] = (1000 * data['road_count'] + data['scaled_suburb'] + data['district_count']).astype(int)

# Display the updated DataFrame with the new column
print(data[['road_count', 'suburb_count', 'scaled_suburb', 'district_count', 'population']].head())

# Optionally, save the updated DataFrame to a new CSV file
data.to_csv('updated_data.csv', index=False)
