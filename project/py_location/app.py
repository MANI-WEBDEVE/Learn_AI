import requests

def get_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        
        # IPinfo.io ke location ko latitude, longitude me convert karein
        lat, lon = data["loc"].split(",")

        # Actual User Location (JavaScript se)
        user_lat = input("Enter user latitude (or press Enter to use detected location): ")
        user_lon = input("Enter user longitude (or press Enter to use detected location): ")

        if user_lat and user_lon:
            lat, lon = user_lat, user_lon

        return {
            "ip": data["ip"],
            "city": data["city"],
            "region": data["region"],
            "country": data["country"],
            "lat": lat,
            "lon": lon,
            "postal": data["postal"],
            "timezone": data["timezone"]
        }
    except Exception as e:
        return {"error": str(e)}
