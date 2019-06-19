import numpy as np
import pandas as pd

from tqdm import tqdm

from math import radians, degrees, cos, sin, asin, sqrt

CENTRE_BOURNEMOUTH = -1.88, 50.72

print("Loading data...")
# Load in the stop_events from the previous stage in the pipeline
stop_events = pd.read_csv(
    "Intermediate_Data/stop_events_with_all_features.csv",
    usecols=["midpoint_lat", "midpoint_lon", "segment_name"],
)

print("\tLoaded")

print("Calculating Distances...")

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


# Find the distance between the midpoints of each segment.
unique_segment_codes = stop_events.groupby("segment_name").first()

distances = pd.DataFrame(index=unique_segment_codes.index)

for name, row in tqdm(unique_segment_codes.iterrows()):

    row_lon = row["midpoint_lon"]
    row_lat = row["midpoint_lat"]

    distances[name] = unique_segment_codes.apply(
        lambda x: haversine(x["midpoint_lon"], x["midpoint_lat"], row_lon, row_lat),
        axis=1,
    )

print("\tCalculated")

print("Writing output file...")

distances.to_csv("Intermediate_Data/intersegment_distances.csv")

print("\tWritten")
