from fastapi import FastAPI, Request
import requests

app = FastAPI()

def get_location(ip):
    """Fetch location details from user's IP"""
    try:
        response = requests.get(f"http://ip-api.com/json/{ip}")
        data = response.json()
        return {
            "ip": ip,
            "city": data.get("city"),
            "region": data.get("regionName"),
            "country": data.get("country"),
            "lat": data.get("lat"),
            "lon": data.get("lon"),
            "isp": data.get("isp")
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def get_user_location(request: Request):
    # Get the real client IP address
    forwarded_for = request.headers.get("X-Forwarded-For")
    print(forwarded_for)
    if forwarded_for:
        user_ip = forwarded_for.split(",")[0]
    else:
        user_ip = request.client.host if request.client else "8.8.8.8"  # Use a public IP for testing
    
    # For local testing, you can use a public IP address
    if user_ip == "127.0.0.1":
        user_ip = "8.8.8.8"  # Replace with a public IP address for testing
    
    location_info = get_location(user_ip)
    return location_info