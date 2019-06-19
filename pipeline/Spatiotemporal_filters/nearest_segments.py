import numpy as np
import pandas as pd

from tqdm import tqdm

from math import radians, degrees, cos, sin, asin, sqrt

CENTRE_BOURNEMOUTH = -1.88, 50.72

print("Loading data...")
# Load in the stop_events from the previous stage in the pipeline
segments = pd.read_csv("Intermediate_Data/intersegment_distances.csv")

print("\tLoaded")

print("Finding Nearest...")

nearest = pd.DataFrame(index=segments["segment_name"])

nearest["nearest10"] = np.empty((nearest.shape[0], 0)).tolist()
nearest["nearest100"] = np.empty((nearest.shape[0], 0)).tolist()
nearest["nearest200"] = np.empty((nearest.shape[0], 0)).tolist()

for index, row in segments.iterrows():

    in_order = np.argsort(row.values[1:])

    name = row[0]

    nearest.at[name, "nearest10"] = segments.columns[in_order[1:11] + 1].values
    nearest.at[name, "nearest100"] = segments.columns[in_order[1:101] + 1].values
    nearest.at[name, "nearest200"] = segments.columns[in_order[1:201] + 1].values

print("\tCalculated")

print("Writing output file...")

nearest.to_csv("Intermediate_Data/nearest_segments.csv")

print("\tWritten")
