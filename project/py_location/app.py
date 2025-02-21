import requests

def get_client_ip():
    """Fetches user's real IP using an external API"""
    try:
        ip_response = requests.get("https://api64.ipify.org?format=json")
        ip_response.raise_for_status()
        user_ip = ip_response.json()["ip"]

        # Now fetch location data for this IP
        location_response = requests.get(f"https://ipinfo.io/{user_ip}/json")
        location_response.raise_for_status()
        
        return location_response.json()

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Get location data
location = get_client_ip()
