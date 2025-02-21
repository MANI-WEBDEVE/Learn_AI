import streamlit as st
import requests
import asyncio

async def load_page():
    st.set_page_config(
        page_title="Location Tracker",
        page_icon="🌍",
        layout="wide"
    )
    
    st.title("Location Tracking System")

  

    # Fetch user location from FastAPI endpoint
    response = requests.get("http://127.0.0.1:8000/")
    if response.status_code == 200:
        location_1 = response.json()
    else:
        st.error("Failed to fetch location data from API")
        return

    # Layout Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User's Actual Location")
        
        latitude = location_1.get("lat", "Unknown")
        longitude = location_1.get("lon", "Unknown")

        st.map(data={"lat": [float(latitude)], "lon": [float(longitude)]})
        st.write(f"Latitude: {latitude}")
        st.write(f"Longitude: {longitude}")
        st.write(f"City: {location_1['city']}")
        st.write(f"Region: {location_1['region']}")
        st.write(f"Country: {location_1['country']}")
        st.write(f"IP: {location_1['ip']}")

if __name__ == "__main__":
    asyncio.run(load_page())