import numpy as np
import pandas as pd
import datetime

from argparse import ArgumentParser
import os.path
from pathlib import Path

from tqdm import tqdm


def filter_rare(stop_events):

    print("Calculating rare segments...")

    segment_counts = stop_events.groupby("segment_name").size()

    filtered_stop_events = stop_events.drop(
        stop_events[
            stop_events["segment_name"].isin(
                segment_counts[segment_counts < 120].index.values
            )
        ].index
    )

    print("\tCalculated")

    return filtered_stop_events


def filter_overtakes(stop_events):

    print("Calculating Overtakes...")

    stop_events = stop_events.assign(
        overtaken_once=False,
        # overtaken_twice=False
    )

    # Find every time any bus did that segment on that day with that public name (route name)
    overtake_opportunities = stop_events.groupby(["date", "segment_name", "publicName"])

    for _, group in tqdm(overtake_opportunities):

        # Sort by when they reached the previous stop
        buses_in_order = group.sort_values("prev_actualArrival", ascending=True)

        # Find the bus after each bus
        one_after = buses_in_order.shift(-1)

        overtaken = buses_in_order[
            buses_in_order["actualArrival"]
            > one_after["actualArrival"] + pd.Timedelta("1 minute")
        ]

        if overtaken.shape[0] == 0:
            continue

        stop_events.loc[overtaken.index, "overtaken_once"] = True

        # two_after = buses_in_order.shift(-2)

        # overtaken_twice = buses_in_order[
        #     buses_in_order["actualArrival"] > two_after["actualArrival"]
        # ]

        # if overtaken_twice.shape[0] == 0:
        #     continue

        # stop_events.loc[overtaken_twice.index, "overtaken_twice"] = True

    print("\tCalculated")

    return stop_events


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

    stop_events = filter_rare(stop_events)

    stop_events = filter_overtakes(stop_events)

    print("Writing output files...")

    # Make sure the folder is there before we write the file to it.
    Path(args.output_filename_once).parent.mkdir(parents=True, exist_ok=True)
    # Path(args.output_filename_twice).parent.mkdir(parents=True, exist_ok=True)

    # We want to write all the rows where overtaken_once is FALSE so we use ~ to invert
    stop_events[~stop_events["overtaken_once"]].drop(
        [
            "overtaken_once",
            # "overtaken_twice"
        ],
        axis=1,
    ).to_csv(args.output_filename_once, index=False)

    # # We want to write all the rows where overtaken_twice is FALSE so we use ~ to invert
    # stop_events[~stop_events["overtaken_twice"]].drop(
    #     ["overtaken_once", "overtaken_twice"], axis=1
    # ).to_csv(args.output_filename_twice, index=False)

    print("\tWritten")
