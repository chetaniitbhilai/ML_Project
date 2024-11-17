import folium
import requests
import polyline

def get_road_route(start_coords, end_coords):
    """
    Get the actual road route between two coordinate points using OSRM.
    
    Args:
        start_coords: Tuple of (latitude, longitude) for starting point
        end_coords: Tuple of (latitude, longitude) for ending point
    Returns:
        Dictionary containing route information including coordinates, distance, and duration
    """
    # OSRM expects coordinates in (longitude, latitude) order
    coords = f"{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
    url = f"http://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=polyline"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data["code"] != "Ok":
            raise Exception("Unable to find route")
            
        route = data["routes"][0]
        decoded_polyline = polyline.decode(route["geometry"])
        distance = route["distance"] / 1000  # Convert to kilometers
        duration = route["duration"] / 60    # Convert to minutes
        
        return {
            "coordinates": decoded_polyline,
            "distance": distance,
            "duration": duration
        }
    except Exception as e:
        print(f"Error getting route: {str(e)}")
        return None

def create_road_map(start_coords, end_coords, save_path='road_map.html'):
    """
    Create a map showing the actual road route between two coordinate points.
    
    Args:
        start_coords: Tuple of (latitude, longitude) for starting point
        end_coords: Tuple of (latitude, longitude) for ending point
        save_path: Path to save the HTML map file
    """
    # Calculate center point for map
    center_lat = (start_coords[0] + end_coords[0]) / 2
    center_lon = (start_coords[1] + end_coords[1]) / 2
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Get road route
    route_info = get_road_route(start_coords, end_coords)
    
    if route_info:
        # Add start marker
        folium.Marker(
            start_coords,
            popup=f'Start Point<br>Coordinates: {start_coords}',
            icon=folium.Icon(color='green', icon='info-sign')
        ).add_to(m)
        
        # Add end marker
        folium.Marker(
            end_coords,
            popup=f'End Point<br>Coordinates: {end_coords}',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Draw route
        folium.PolyLine(
            locations=route_info["coordinates"],
            weight=3,
            color='blue',
            opacity=0.8,
            popup=f'Distance: {route_info["distance"]:.2f} km<br>'
                  f'Duration: {route_info["duration"]:.1f} min'
        ).add_to(m)
        
        # Fit map bounds to show entire route
        m.fit_bounds([start_coords, end_coords])
        
        # Save map
        m.save(save_path)
        return route_info
    return None

# Example usage
if __name__ == "__main__":
    # Example coordinates
    start_coords=input("Enter start coordinates: ").split(",")
    end_coords=input("Enter end coordinates: ").split(",")
    # Create route map
    route_info = create_road_map(start_coords, end_coords, 'route_map.html')
    
    if route_info:
        print("\nRoute Details:")
        print(f"Start Coordinates: {start_coords}")
        print(f"End Coordinates: {end_coords}")
        print(f"Distance: {route_info['distance']:.2f} km")
        print(f"Estimated duration: {route_info['duration']:.1f} minutes")
