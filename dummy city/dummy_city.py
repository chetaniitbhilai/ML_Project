import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import networkx as nx
class DummyCity:
    def __init__(self, size=50):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.facilities = {
            0: 'Empty',
            1: 'Road',
            2: 'House',
            3: 'Hospital',
            4: 'School',
            5: 'Park',
            6: 'Shop',
            7: 'Office',
            8: 'Industrial',
            9: 'Water',
            10: 'Forest'
        }
        self.population_density = np.zeros((size, size))
        self.activity_centers = []

    def add_activity_centers(self, num_centers=5):
        for _ in range(num_centers):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            self.activity_centers.append((x, y))
            self.grid[x-2:x+3, y-2:y+3] = 6  # Shops
            self.grid[x-1:x+2, y-1:y+2] = 7  # Offices
            self.grid[x, y] = 3  # Hospital at the center

    def add_terrain(self):
        num_water = random.randint(1, 3)
        for _ in range(num_water):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            size = random.randint(5, 10)
            self.grid[max(0, x-size):min(self.size, x+size),
                      max(0, y-size):min(self.size, y+size)] = 9

        num_forests = random.randint(2, 5)
        for _ in range(num_forests):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            size = random.randint(5, 15)
            forest = (self.grid[max(0, x-size):min(self.size, x+size),
                                max(0, y-size):min(self.size, y+size)] == 0)
            self.grid[max(0, x-size):min(self.size, x+size),
                      max(0, y-size):min(self.size, y+size)][forest] = 10

    def add_roads(self):
        # Create a graph to connect activity centers
        G = nx.Graph()
        for i, (x1, y1) in enumerate(self.activity_centers):
            for j, (x2, y2) in enumerate(self.activity_centers):
                if i != j:
                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    G.add_edge(i, j, weight=distance)

        # Generate minimum spanning tree (MST) for primary roads
        mst = nx.minimum_spanning_tree(G)
        for u, v in mst.edges():
            x1, y1 = self.activity_centers[u]
            x2, y2 = self.activity_centers[v]
            self.connect_points(x1, y1, x2, y2, road_type='primary')

        # Add secondary roads in select areas (near activity centers and main zones)
        self.add_secondary_roads()

        # Add very few tertiary roads, keeping them minimal
        self.add_tertiary_roads()

    def connect_points(self, x1, y1, x2, y2, road_type='primary'):
        """
        Connect two points (activity centers) with a straight road.
        Road type defines the width or priority of the road.
        """
        if road_type == 'primary':
            width = 2  # Primary roads are moderately wide
        else:
            width = 1  # Secondary and tertiary roads are narrower

        # Connect points with a straight road (L-shaped or Manhattan distance)
        if random.choice([True, False]):
            self.grid[x1, min(y1, y2):max(y1, y2)] = 1  # Horizontal segment
            self.grid[min(x1, x2):max(x1, x2), y2-width:y2+width] = 1  # Vertical segment
        else:
            self.grid[min(x1, x2):max(x1, x2), y1-width:y1+width] = 1  # Vertical segment
            self.grid[x2, min(y1, y2):max(y1, y2)] = 1  # Horizontal segment

    def add_secondary_roads(self):
        """
        Add secondary roads connecting key areas but reducing their overall number.
        """
        spacing = 15  # More sparse secondary road spacing for a realistic feel
        for i in range(0, self.size, spacing):
            self.grid[i, :] = 1  # Horizontal secondary roads
            self.grid[:, i] = 1  # Vertical secondary roads

    def add_tertiary_roads(self):
        """
        Add only a few tertiary roads within neighborhoods.
        These roads are added in areas near houses and activity centers.
        """
        num_local_roads = 15  # Reduced number of tertiary roads for realism
        for _ in range(num_local_roads):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            length = random.randint(5, 10)
            if random.choice([True, False]):
                self.grid[x, max(0, y-length):min(self.size, y+length)] = 1  # Horizontal local road
            else:
                self.grid[max(0, x-length):min(self.size, x+length), y] = 1  # Vertical local road

    def add_facilities(self, num_hospitals=10):
        for center in self.activity_centers:
            x, y = center
            for _ in range(100):
                attempts = 0
                while attempts < 100:
                    dx, dy = random.randint(-10, 10), random.randint(-10, 10)
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[nx, ny] == 0:
                        self.grid[nx, ny] = 2  # House
                        self.update_population_density(nx, ny)
                        break
                    attempts += 1

            self.grid[x+5:x+8, y+5:y+8] = 5  # Park
            self.grid[x-5:x-3, y-5:y-3] = 4  # School

        # Add additional hospitals
        for _ in range(num_hospitals):
            attempts = 0
            while attempts < 100:
                x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
                if self.grid[x, y] == 0:
                    self.grid[x-2:x+3, y-2:y+3] = 3  # Hospital occupies a 5x5 area
                    break
                attempts += 1

    def update_population_density(self, x, y, radius=5):
        x_min, x_max = max(0, x-radius), min(self.size, x+radius+1)
        y_min, y_max = max(0, y-radius), min(self.size, y+radius+1)
        xx, yy = np.ogrid[x_min:x_max, y_min:y_max]
        distances = np.sqrt((xx-x)**2 + (yy-y)**2)
        density = np.exp(-distances**2 / (2 * (radius/3)**2))
        self.population_density[x_min:x_max, y_min:y_max] += density

    def visualize(self):
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        colors = ['#FFFFFF', '#808080', '#FFA07A', '#FF0000', '#FFFF00', '#00FF00',
                  '#FFA500', '#ADD8E6', '#A52A2A', '#0000FF', '#008000']
        labels = ['Empty', 'Road', 'House', 'Hospital', 'School', 'Park', 'Shop',
                  'Office', 'Industrial', 'Water', 'Forest']
        cmap = plt.cm.colors.ListedColormap(colors)
        bounds = np.arange(len(colors) + 1) - 0.5
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(self.grid, cmap=cmap, norm=norm)

        plt.colorbar(img, ticks=np.arange(len(labels)), boundaries=bounds, format='%1i')
        img.colorbar.set_ticks(np.arange(len(labels)))
        img.colorbar.set_ticklabels(labels)
        plt.title('City Layout')

        plt.subplot(1, 2, 2)
        plt.imshow(self.population_density, cmap='viridis')
        plt.colorbar()
        plt.title('Population Density')

        plt.show()
# Usage
city = DummyCity(size=100)
city.add_activity_centers(num_centers=10)
city.add_terrain()
city.add_roads()
city.add_facilities()
city.visualize()
