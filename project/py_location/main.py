import streamlit as st
import app  # Import the corrected app.py file
# import api  # Import API module
from api.locationapi import Item 
import api.locationapi as api
# Configure Streamlit Page
st.set_page_config(
    page_title="Location Tracker",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Location Tracking System")

# Fetch client IP location
location_1 = app.get_client_ip()

# Ensure valid data
if "error" in location_1:
    st.error(f"Error Fetching IP: {location_1['error']}")
else:
    st.success("Location data retrieved successfully!")

    # Convert location data to Pydantic model
    location_item = Item(**location_1)
    
    # Store location in MongoDB
    response = api.create_loc(location_item)
    st.write("📍 **Location Saved in Database!**")

    # Display Location Data
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Location Details")
        loca = list(location_1['loc'].split(","))
        latitude, longitude = float(loca[0]), float(loca[1])

        st.map(data={"lat": [latitude], "lon": [longitude]})
        st.write(f"**City:** {location_1['city']}")
        st.write(f"**Region:** {location_1['region']}")
        st.write(f"**Country:** {location_1['country']}")
        st.write(f"**IP:** {location_1['ip']}")
        st.write(f"**Postal Code:** {location_1['postal']}")
        st.write(f"**Timezone:** {location_1['timezone']}")

    with col2:
        st.subheader("📜 Location History")
        if "locations" not in st.session_state:
            st.session_state.locations = []

        if latitude != 0.0 and longitude != 0.0:
            location = {"latitude": latitude, "longitude": longitude, "timestamp": st.time_input("Timestamp")}
            if st.button("Save Location"):
                st.session_state.locations.append(location)

        if st.session_state.locations:
            for loc in st.session_state.locations:
                st.write(f"Lat: {loc['latitude']}, Long: {loc['longitude']}, Time: {loc['timestamp']}")
