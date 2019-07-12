import numpy as np
import pandas as pd
import datetime

from argparse import ArgumentParser
import os.path
from pathlib import Path

from tqdm import tqdm


def filter_3sigma(stop_events):

    print("Calculating 3 sigma...")

    stop_events["prev_actualDeparture"] = stop_events["prev_actualDeparture"].astype(
        "datetime64[ns]"
    )
    stop_events["actualArrival"] = stop_events["actualArrival"].astype("datetime64[ns]")

    stop_events["segment_duration"] = (
        stop_events.actualArrival - stop_events.prev_actualDeparture
    ).astype("timedelta64[s]")

    means = stop_events.groupby("segment_code")["segment_duration"].mean()

    std = stop_events.groupby("segment_code")["segment_duration"].std()

    means = means.rename("means")
    std = std.rename("std")

    stop_events = stop_events.merge(
        means.to_frame(), left_on="segment_code", right_index=True
    )
    stop_events = stop_events.merge(
        std.to_frame(), left_on="segment_code", right_index=True
    )

    stop_events["outliers"] = False
    stop_events.loc[
        (
            stop_events["segment_duration"]
            > (stop_events["means"] + 3 * stop_events["std"])
        )
        | (
            stop_events["segment_duration"]
            < (stop_events["means"] - 3 * stop_events["std"])
        ),
        "outliers",
    ] = True

    return stop_events[~stop_events["outliers"]].drop(columns=["means", "std"])


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, "r")  # return an open file handle


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
        "-once",
        dest="output_filename_once",
        required=True,
        help="overtaken once file name and path to write to",
        metavar="FILE",
    )

    # parser.add_argument(
    #     "-twice",
    #     dest="output_filename_twice",
    #     required=True,
    #     help="overtaken twice file name and path to write to",
    #     metavar="FILE",
    # )

    args = parser.parse_args()

    print("Loading data...")
    # Load in the stop_events from the previous stage in the pipeline
    stop_events = pd.read_csv(args.input_filename, parse_dates=[1, 5, 6, 18, 19])

    # Force it to treat all the times as actual times.
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

    stop_events = filter_3sigma(stop_events)

    print("Writing output files...")

    # Make sure the folder is there before we write the file to it.
    Path(args.output_filename_once).parent.mkdir(parents=True, exist_ok=True)
    # Path(args.output_filename_twice).parent.mkdir(parents=True, exist_ok=True)

    # We want to write all the rows where overtaken_once is FALSE so we use ~ to invert
    stop_events.to_csv(args.output_filename_once, index=False)

    print("\tWritten")
