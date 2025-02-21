import requests

def get_user_ip():
    try:
        response = requests.get("https://api64.ipify.org?format=json")
        ip_data = response.json()
        return ip_data["ip"]
    except Exception as e:
        print("Error fetching IP:", e)
        return None

def get_location(ip):
    try:
        response = requests.get(f"https://ipinfo.io/{ip}/json")
        return response.json()
    except Exception as e:
        print("Error fetching location:", e)
        return None

user_ip = get_user_ip()
print(f'user_ip:{user_ip}')
if user_ip:
    location_data = get_location(user_ip)
    print(f'terminal:{location_data}')
    def user_info():
        return location_data

else:
    print("Could not get user IP")
