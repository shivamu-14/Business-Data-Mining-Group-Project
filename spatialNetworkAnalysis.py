# Import necessary libraries
import webbrowser
import networkx as nx
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import geopandas as gpd
from geopy.distance import geodesic

df = pd.read_csv("uber.csv")
# Filtering out zero values for coordinates
df = df[(df['Restaurant_latitude'] != 0) & (df['Restaurant_longitude'] != 0) &
        (df['Delivery_location_latitude'] != 0) & (df['Delivery_location_longitude'] != 0)]

# Cluster Analysis - KMeans Clustering with multiple colors
location_data = df[['Restaurant_latitude', 'Restaurant_longitude']]
kmeans = KMeans(n_clusters=5, random_state=42)
df['location_cluster'] = kmeans.fit_predict(location_data)

# data for heatmap
heat_data = [[row['Restaurant_latitude'], row['Restaurant_longitude']] for _, row in df.iterrows()]

#base map centered on average location
m = folium.Map(location=[df['Restaurant_latitude'].mean(), df['Restaurant_longitude'].mean()],
               zoom_start=10, control_scale=True)

# Overlaying the spatial heatmap
HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'orange', 1: 'red'}).add_to(m)

# Network Analysis and adding KMeans clusters as markers
marker_cluster = MarkerCluster().add_to(m)
G = nx.Graph()

# colors for clusters
cluster_colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF']

for _, row in df.iterrows():
    #nodes for the restaurant and delivery location
    restaurant_node = (row['Restaurant_latitude'], row['Restaurant_longitude'])
    delivery_node = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])

    G.add_node(restaurant_node)
    G.add_node(delivery_node)

    #edge (route) between restaurant and delivery location
    G.add_edge(restaurant_node, delivery_node, weight=row['DeliveryDistance'])

    #color based on the cluster
    cluster_color = cluster_colors[row['location_cluster']]

    #restaurant and delivery markers to the map
    folium.Marker(location=restaurant_node, popup="Restaurant", icon=folium.Icon(color='blue')).add_to(marker_cluster)
    folium.Marker(location=delivery_node, popup="Delivery", icon=folium.Icon(color='green')).add_to(marker_cluster)

    #route line between restaurant and delivery location
    folium.PolyLine(locations=[restaurant_node, delivery_node], color=cluster_color, weight=2.5, opacity=0.7).add_to(m)

#combined map with spatial and network analysis
m.save("combined_spatial_network_map.html")
print("Combined spatial and network analysis map saved as 'combined_spatial_network_map.html'.")

# Converting locations to geometry points
df['restaurant_coords'] = df.apply(lambda row: Point(row['Restaurant_longitude'], row['Restaurant_latitude']), axis=1)
df['delivery_coords'] = df.apply(
    lambda row: Point(row['Delivery_location_longitude'], row['Delivery_location_latitude']), axis=1)

# GeoDataFrame with restaurant coordinates as geometry
gdf = gpd.GeoDataFrame(df, geometry='restaurant_coords')

#distances using geopy for more accurate distances in km
gdf['distance_km'] = gdf.apply(lambda row: geodesic((row['Restaurant_latitude'], row['Restaurant_longitude']),
                                                    (row['Delivery_location_latitude'],
                                                     row['Delivery_location_longitude'])).km, axis=1)

#distance distribution
plt.figure(figsize=(10, 6))
sns.histplot(gdf['distance_km'], bins=30, kde=True)
plt.title('Distance Distribution of Deliveries')
plt.xlabel('Distance (km)')
plt.ylabel('Frequency of orders')
plt.show()

