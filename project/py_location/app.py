import geocoder
import folium
import requests
# Use 'me' to get the current IP address of the machine running the code
g = geocoder.ip('39.39.124.59')
my_address = g.latlng

def get_location():
    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    return data

location = get_location()
# print(f"IP: {location['ip']}")
# print(f"City: {location['city']}")
# print(f"Region: {location['region']}")
# print(f"Country: {location['country']}")
print(f"Location (Lat, Long): {location['loc']}")

if my_address:
    my_map = folium.Map(location=my_address, zoom_start=12)
    folium.CircleMarker(location=my_address, radius=50, popup="Your Location").add_to(my_map)
    folium.Marker(location=my_address, popup="Yorkshire").add_to(my_map)
    my_map.save("my_map.html")
    # print(my_address)
else:
    print("Failed to get the location.")