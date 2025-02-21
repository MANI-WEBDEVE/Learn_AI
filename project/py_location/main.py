import streamlit as st
import app
import api.locationapi as api
import asyncio
from api.locationapi import Item

async def load_page():
    st.set_page_config(
        page_title="Location Tracker",
        page_icon="🌍",
        layout="wide"
    )
    location_1 = app.user_info()
    st.write(f'location_1: {location_1}')
    st.title("Location Tracking System")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Convert location_1 dictionary to Item object
    if location_1 is not None and all(key in location_1 for key in ["ip", "region", "country", "loc", "postal", "city", "timezone"]):
        location_item = Item(**location_1)
        lloop = await api.create_loc(location_item)
        st.write(f'loop: {lloop}')

        with col1:
            # Location input fields
            loca = list(location_1['loc'].split(","))
            
            st.subheader("Enter Location Details")
            latitude = float(loca[0])
            longitude = float(loca[1])
            st.map(data={"lat": [latitude], "lon": [longitude]})
            st.write(f"City: {location_1['city']}")
            st.write(f"Region: {location_1['region']}")
            st.write(f"Country: {location_1['country']}")
            st.write(f"IP: {location_1['ip']}")
            st.write(f"Postal Code: {location_1['postal']}")
            st.write(f"Timezone: {location_1['timezone']}")
    else:
        st.error("Location information is incomplete or missing.")
        
    with col2:
        # Location history section
        st.subheader("Location History")
        if "locations" not in st.session_state:
            st.session_state.locations = []
            
        if latitude != 0.0 and longitude != 0.0:
            location = {"latitude": latitude, "longitude": longitude, "timestamp": st.time_input("Timestamp")}
            if st.button("Save Location"):
                st.session_state.locations.append(location)
        # Display saved locations
        if st.session_state.locations:
            for loc in st.session_state.locations:
                st.write(f"Lat: {loc['latitude']}, Long: {loc['longitude']}, Time: {loc['timestamp']}")



# # Add footer at the bottom of the page
# st.markdown("""
# ---
# Created MANI🐱‍💻
# """)
if __name__ == "__main__":
    asyncio.run(load_page())