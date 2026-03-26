


import zipfile 
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import pydeck as pdk
from shapely.geometry import Point

from src.component.inference import (
    load_model_from_registry,
    load_batch_of_features_from_store,
    get_model_predictions
)
from src.component.inference import (
    load_predictions_from_store,
    load_batch_of_features_from_store
)

from paths import DATA_DIR
from plot import plot_one_sample

st.set_page_config(layout="wide")

# title
current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')
st.title(f'Electricity Demand Prediction ⚡ — Bangalore, India')
st.header(f'{current_date} UTC')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 6

# Define Karnataka / Bangalore Electricity Zones with approximate centers
# Based on BESCOM (Bangalore) and other Karnataka ESCOM regions
karnataka_zones = {
     0: {'name': 'BESCOM North (Bangalore North)', 'lat': 13.0358, 'lon': 77.5970},
     1: {'name': 'BESCOM South (Bangalore South)', 'lat': 12.9000, 'lon': 77.5800},
     2: {'name': 'BESCOM East (Whitefield/KR Puram)', 'lat': 12.9698, 'lon': 77.7500},
     3: {'name': 'BESCOM West (Rajajinagar/Peenya)', 'lat': 12.9900, 'lon': 77.5200},
     4: {'name': 'MESCOM (Mysore Region)', 'lat': 12.2958, 'lon': 76.6394},
     5: {'name': 'CESC (Mangalore/Coastal)', 'lat': 12.9141, 'lon': 74.8560},
     6: {'name': 'HESCOM (Hubli-Dharwad)', 'lat': 15.3647, 'lon': 75.1240},
     7: {'name': 'GESCOM (Gulbarga/Kalaburagi)', 'lat': 17.3297, 'lon': 76.8343},
}

# Create a GeoDataFrame from the Karnataka zones
def create_karnataka_geo_df():
    zones = []
    for zone_id, info in karnataka_zones.items():
        zones.append({
            'zone_id': zone_id,
            'name': info['name'],
            'latitude': info['lat'],
            'longitude': info['lon']
        })
    df = pd.DataFrame(zones)
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitude, df.latitude)
    )
    return gdf


with st.spinner(text="Creating Karnataka zones data"):
    geo_df = create_karnataka_geo_df()
    st.sidebar.write('✅ Karnataka zones data created')
    progress_bar.progress(1/N_STEPS)

with st.spinner(text="Fetching model predictions from the store"):
    predictions_df = load_predictions_from_store(
        from_date=current_date - timedelta(hours=1),
        to_date=current_date
    )
    st.sidebar.write('✅ Model predictions arrived')
    progress_bar.progress(2/N_STEPS)

from datetime import timedelta

# Check if predictions for the current hour are available
next_hour_predictions_ready = not predictions_df[predictions_df.date == current_date].empty
prev_hour_predictions_ready = not predictions_df[predictions_df.date == (current_date - timedelta(hours=1))].empty

if next_hour_predictions_ready:
    # Predictions for the next hour are already available
    predictions_df = predictions_df[predictions_df.date == current_date]

else:
    # Attempt to fetch predictions for the next hour
    with st.spinner(text="Fetching batch of data"):
        features = load_batch_of_features_from_store(current_date)

    with st.spinner(text="Loading ML model from registry"):
        model = load_model_from_registry()

    with st.spinner(text="Computing model predictions"):
        predictions = get_model_predictions(model, features)

    # Update predictions availability after fetching
    next_hour_predictions_ready = not predictions_df[predictions_df.date == current_date].empty

    if next_hour_predictions_ready:
        predictions_df = predictions_df[predictions_df.date == current_date]

    elif prev_hour_predictions_ready:
        # If next-hour predictions are still unavailable, use previous hour predictions
        predictions_df = predictions_df[predictions_df.date == (current_date - timedelta(hours=1))]
        current_date = current_date - timedelta(hours=1)
        st.subheader('⚠️ The most recent data is not yet available. Using last hour predictions')

    else:
        raise Exception('Features are not available for the last 2 hours. Is your feature pipeline up and running? 🤔')


with st.spinner(text="Preparing data to plot"):
    def pseudocolor(val, minval, maxval, startcolor, stopcolor, alpha=300):
        f = float(val - minval) / (maxval - minval)
        rgb = tuple(int(f * (b - a) + a) for a, b in zip(startcolor, stopcolor))
        return rgb + (alpha,)
    # Merge your data
    df = pd.merge(geo_df, predictions_df,
                  right_on='sub_region_code',
                  left_on='zone_id',
                  how='inner')
    
    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    
    # Scale the radius based on predicted demand
    scaling_factor = 5 # Adjust this factor as needed for your visualization
    df['radius'] = df['predicted_demand'] * scaling_factor
    progress_bar.progress(5 / N_STEPS)

with st.spinner(text="Generating Karnataka Zones Map"):
    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=12.9716,   # Centered on Bangalore
        longitude=77.5946,
        zoom=6,
        max_zoom=16,
        pitch=45,
        bearing=0
    )

    geojson = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_radius="radius",         # Use the dynamic radius here
        get_fill_color=[255, 0, 0],   # Use the computed fill colors
        pickable=True
    )

    tooltip = {
        "html": "<b>Zone:</b> [{zone_id}] {name} <br /> <b>Predicted demand (MW):</b> {predicted_demand}"
    }

    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    st.pydeck_chart(r)

    progress_bar.progress(4/N_STEPS)

with st.spinner(text="Fetching batch of features used in the last run"):
    features_df = load_batch_of_features_from_store(current_date)
    st.sidebar.write('✅ Inference features fetched from the store')
    progress_bar.progress(5/N_STEPS)

with st.spinner(text="Plotting time-series data"):
   
    predictions_df = df

    row_indices = np.argsort(predictions_df['predicted_demand'].values)[::-1]
    n_to_plot = 6

    for row_id in row_indices[:n_to_plot]:
        location_id = predictions_df['zone_id'].iloc[row_id]
        location_name = predictions_df['name'].iloc[row_id]
        st.header(f'Zone ID: {location_id} - {location_name}')

        prediction = predictions_df['predicted_demand'].iloc[row_id]
        st.metric(label="Predicted demand (MW)", value=int(prediction))
        
        fig = plot_one_sample(
            example_id=row_id,
            features=features_df,
            targets=predictions_df['predicted_demand'],
            predictions=pd.Series(predictions_df['predicted_demand']),
            display_title=False,
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)
        
    progress_bar.progress(6/N_STEPS)



