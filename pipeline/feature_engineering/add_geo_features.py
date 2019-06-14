import numpy as np
import pandas as pd
import datetime

from tqdm import tqdm

from math import radians, degrees, cos, sin, asin, sqrt

CENTRE_BOURNEMOUTH = -1.88, 50.72

print("Loading data...")
# Load in the stop_events from the previous stage in the pipeline
stop_events = pd.read_csv(
    "Intermediate_Data/stop_events.csv", parse_dates=[1, 5, 6, 18, 19]
)


# Force it to treat all the times as actual times.
stop_events["aimedArrival"] = stop_events["aimedArrival"].astype("datetime64[ns]")
stop_events["aimedDeparture"] = stop_events["aimedDeparture"].astype("datetime64[ns]")
stop_events["prev_aimedArrival"] = stop_events["prev_aimedArrival"].astype(
    "datetime64[ns]"
)
stop_events["prev_aimedDeparture"] = stop_events["prev_aimedDeparture"].astype(
    "datetime64[ns]"
)
stop_events["prev_actualArrival"] = stop_events["prev_actualArrival"].astype(
    "datetime64[ns]"
)
stop_events["prev_actualDeparture"] = stop_events["prev_actualDeparture"].astype(
    "datetime64[ns]"
)

print("\tLoaded")

print("Dropping nans...")
# Drop any rows with nan or empty sections.
stop_events = stop_events.dropna(axis=0)
print("\tDropped")

print("Calculating Distances...")


# Load the stops data which contains the lat and lon for each stop
stops = pd.read_csv("Trapeze_Data/Stops.csv")
stops = stops.set_index("stopCode")


# From: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


# Go through each segment code and work out the straight line distance and midpoint for the segment
# Technically using segment code causes some duplication as there is the stop point and no stop point
# version but it's still faster than re-calculating the unique pairs and makes it easy to reference
# it again to assign the values.
unique_segment_codes = pd.unique(stop_events["segment_code"])

stop_events = stop_events.assign(
    line_distance=0,
    midpoint_lat=0,
    midpoint_lon=0,
    to_centre_dist=0,
    direction=0,
    direction_degrees=0,
)

for segment_code in tqdm(unique_segment_codes):

    codes = segment_code.split("_")

    from_code = codes[0]
    to_code = codes[1]

    if from_code == to_code:
        continue

    from_coords = stops.loc[from_code].values
    to_coords = stops.loc[to_code].values

    mid_coords = (from_coords + to_coords) / 2

    line_distance = haversine(*from_coords, *to_coords)
    mid_centre_distance = haversine(*CENTRE_BOURNEMOUTH, *mid_coords)

    from_centre_dist = haversine(*from_coords, *CENTRE_BOURNEMOUTH)
    to_centre_distance = haversine(*to_coords, *CENTRE_BOURNEMOUTH)

    direction = (from_centre_dist - to_centre_distance) / line_distance

    direction_degrees = degrees(np.arccos(direction)) - 90

    # print(line_distance)
    # print(to_centre_distance)

    stop_events.loc[
        stop_events["segment_code"] == segment_code,
        [
            "line_distance",
            "midpoint_lat",
            "midpoint_lon",
            "to_centre_dist",
            "direction",
            "direction_degrees",
        ],
    ] = (
        line_distance,
        mid_coords[0],
        mid_coords[1],
        mid_centre_distance,
        direction,
        direction_degrees,
    )

print("\tCalculated")

print("Writing output file...")

stop_events.to_csv("Intermediate_Data/stop_events_with_geo_features.csv", index=False)

print("\tWritten")
