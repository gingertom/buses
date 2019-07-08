import numpy as np
import pandas as pd
import datetime

from argparse import ArgumentParser
import os.path
from pathlib import Path

from tqdm import tqdm

import feather

from math import radians, degrees, cos, sin, asin, sqrt

CENTRE_BOURNEMOUTH = -1.88, 50.72

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


def add_geo_features(stop_events, centre_coords):

    print("Calculating Distances...")

    # Load the stops data which contains the lat and lon for each stop
    stops = pd.read_csv("Trapeze_Data/Stops.csv")
    stops = stops.set_index("stopCode")

    # Go through each segment code and work out the straight line distance and midpoint for the segment
    # Technically using segment code causes some duplication as there is the stop point and no stop point
    # version but it's still faster than re-calculating the unique pairs and makes it easy to reference
    # it again to assign the values.
    unique_segment_names = pd.unique(stop_events["segment_name"])

    stop_events = stop_events.assign(
        line_distance=0,
        midpoint_lat=0,
        midpoint_lon=0,
        to_centre_dist=0,
        direction=0,
        direction_degrees=0,
        clock_direction_degrees=0,
    )

    for segment_name in tqdm(unique_segment_names):

        codes = segment_name.split("_")

        from_code = codes[0]
        to_code = codes[1]

        if from_code == to_code:
            continue

        from_coords = stops.loc[from_code].values
        to_coords = stops.loc[to_code].values

        mid_coords = (from_coords + to_coords) / 2

        line_distance = haversine(*from_coords, *to_coords)
        mid_centre_distance = haversine(*centre_coords, *mid_coords)

        from_centre_dist = haversine(*from_coords, *centre_coords)
        to_centre_distance = haversine(*to_coords, *centre_coords)

        direction = (from_centre_dist - to_centre_distance) / line_distance

        direction_degrees = degrees(np.arccos(direction)) - 90

        adjacent = haversine(mid_coords[0], centre_coords[1], *centre_coords)

        if mid_coords[0] < CENTRE_BOURNEMOUTH[0]:
            adjacent *= -1

        clock_direction_degrees = 180 - degrees(
            np.arccos(adjacent / mid_centre_distance)
        )

        # print(line_distance)
        # print(to_centre_distance)

        stop_events.loc[
            stop_events["segment_name"] == segment_name,
            [
                "line_distance",
                "midpoint_lon",
                "midpoint_lat",
                "to_centre_dist",
                "direction",
                "direction_degrees",
                "clock_direction_degrees",
            ],
        ] = (
            line_distance,
            mid_coords[0],
            mid_coords[1],
            mid_centre_distance,
            direction,
            direction_degrees,
            clock_direction_degrees,
        )

    print("\tCalculated")

    return stop_events


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg  # return filename


if __name__ == "__main__":

    parser = ArgumentParser(description="add geo features")
    parser.add_argument(
        "-i",
        dest="input_filename",
        required=True,
        help="input csv file from a data_reader",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
    )

    parser.add_argument(
        "-o",
        dest="output_filename",
        required=True,
        help="file name and path to write to",
        metavar="FILE",
    )

    args = parser.parse_args()

    print("Loading data...")
    # Load in the stop_events from the previous stage in the pipeline
    stop_events = feather.read_dataframe(args.input_filename)
    stop_events = stop_events.set_index(stop_events.columns[0])

    print("\tLoaded")

    # print("Dropping nans...")
    # # Drop any rows with nan or empty sections.
    # stop_events = stop_events.dropna(axis=0)
    # print("\tDropped")

    stop_events = add_geo_features(stop_events, CENTRE_BOURNEMOUTH)

    print("Writing output file...")

    # Make sure the folder is there before we write the file to it.
    Path(args.output_filename).parent.mkdir(parents=True, exist_ok=True)

    stop_events = stop_events.reset_index()

    stop_events.to_feather(args.output_filename)

    print("\tWritten")
