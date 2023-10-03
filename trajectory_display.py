import folium
import h3
import json
import pandas as pd

#we begin by taking a trajectory from train_clean_small.json
#we take the first trajectory of the first taxi

#we read the data
with open('data/train_clean_small.json', 'r') as openfile:
     
        # Reading from json file
        json_loaded = json.load(openfile)

#we put them in a dataframe
data_clean = pd.DataFrame(data=json_loaded)

#we take the first trajectory of the first taxi
trajectory = data_clean['POLYLINE'][0]

#Now we want to display the trajectory on a map

map_center = [41.157943, -8.629105] # Porto coordinates

#we create the map
m = folium.Map(location=map_center, zoom_start=13)

resolution = 10
center_lat = 41.157943
center_lon = -8.629105
hexagons = h3.h3_set_to_multi_polygon([h3.geo_to_h3(center_lat, center_lon, resolution)])
#the line above is to create the hexagons on the map

#now we want to display the trajectory on the map and we want to display the hexagons on the map too
#we begin by displaying the hexagons
folium.GeoJson(hexagons).add_to(m)

#we display the trajectory
folium.PolyLine(trajectory, color="red", weight=2.5, opacity=1).add_to(m)

#this code works when the trajectory is a list of lists of coordinates in the form [lat, lon], so please make sure that the trajectory is in this form
#The porto dataset is in this form only after the preprocessing step which is done elsewhere
