import numpy as np
import pandas as pd
import datetime

from argparse import ArgumentParser
import os.path
from pathlib import Path


def pivot(stop_events):

    print("Pivoting data...")

    stop_events["arrival_5mins"] = stop_events["actualArrival"].dt.round("5min")
    stop_events["arrival_10mins"] = stop_events["actualArrival"].dt.round("10min")
    stop_events["arrival_1hour"] = stop_events["date"] + pd.to_timedelta(
        stop_events["arrival_hour"].values, unit="h"
    )

    pivoted_5mins_code_hour_day = stop_events.pivot_table(
        index="arrival_5mins",
        columns="segment_code",
        values="diff_percent_full_segment_and_median_by_segment_code_and_hour_and_day",
        aggfunc=np.mean,
    )

    pivoted_5mins_code = stop_events.pivot_table(
        index="arrival_5mins",
        columns="segment_code",
        values="diff_percent_full_segment_and_median_by_segment_code",
        aggfunc=np.mean,
    )

    pivoted_10mins_code_hour_day = stop_events.pivot_table(
        index="arrival_10mins",
        columns="segment_code",
        values="diff_percent_full_segment_and_median_by_segment_code_and_hour_and_day",
        aggfunc=np.mean,
    )

    pivoted_10mins_code = stop_events.pivot_table(
        index="arrival_10mins",
        columns="segment_code",
        values="diff_percent_full_segment_and_median_by_segment_code",
        aggfunc=np.mean,
    )

    pivoted_1hour_code_hour_day = stop_events.pivot_table(
        index="arrival_1hour",
        columns="segment_code",
        values="diff_percent_full_segment_and_median_by_segment_code_and_hour_and_day",
        aggfunc=np.mean,
    )

    pivoted_1hour_code = stop_events.pivot_table(
        index="arrival_1hour",
        columns="segment_code",
        values="diff_percent_full_segment_and_median_by_segment_code",
        aggfunc=np.mean,
    )

    print("\tpivoted")

    return (
        pivoted_5mins_code_hour_day,
        pivoted_5mins_code,
        pivoted_10mins_code_hour_day,
        pivoted_10mins_code,
        pivoted_1hour_code_hour_day,
        pivoted_1hour_code,
    )


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

    # parser.add_argument(
    #     "-o",
    #     dest="output_filename",
    #     required=True,
    #     help="file name and path to write to",
    #     metavar="FILE",
    # )

    args = parser.parse_args()

    print("Loading data...")
    # Load in the stop_events from the previous stage in the pipeline
    stop_events = pd.read_csv(args.input_filename, parse_dates=[1, 5, 6, 18, 19])

    # Force it to treat all the times as actual times.
    stop_events["aimedArrival"] = stop_events["aimedArrival"].astype("datetime64[ns]")
    stop_events["aimedDeparture"] = stop_events["aimedDeparture"].astype(
        "datetime64[ns]"
    )
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

    stop_events["date"] = stop_events["date"].astype("datetime64[ns]")

    print("\tLoaded")

    print("Dropping nans...")
    # Drop any rows with nan or empty sections.
    stop_events = stop_events.dropna(axis=0)
    print("\tDropped")

    pivoted_5mins_code_hour_day, pivoted_5mins_code, pivoted_10mins_code_hour_day, pivoted_10mins_code, pivoted_1hour_code_hour_day, pivoted_1hour_code = pivot(
        stop_events
    )

    print("Writing output file...")

    # Make sure the folder is there before we write the file to it.
    to_write_path = Path(str(args.input_filename)).parent

    path_5mins = to_write_path / Path("5mins")

    path_5mins.mkdir(parents=True, exist_ok=True)

    pivoted_5mins_code_hour_day.to_csv(
        str(path_5mins)
        + "/diff_percent_from_code_hour_day_-_full_segment_time_series.csv"
    )

    pivoted_5mins_code.to_csv(
        str(path_5mins) + "/diff_percent_from_code_-_full_segment_time_series.csv"
    )

    path_10mins = to_write_path / Path("10mins")

    path_10mins.mkdir(parents=True, exist_ok=True)

    pivoted_10mins_code_hour_day.to_csv(
        str(path_10mins)
        + "/diff_percent_from_code_hour_day_-_full_segment_time_series.csv"
    )

    pivoted_10mins_code.to_csv(
        str(path_10mins) + "/diff_percent_from_code_-_full_segment_time_series.csv"
    )

    path_1hour = to_write_path / Path("1hour")

    path_1hour.mkdir(parents=True, exist_ok=True)

    pivoted_1hour_code_hour_day.to_csv(
        str(path_1hour)
        + "/diff_percent_from_code_hour_day_-_full_segment_time_series.csv"
    )

    pivoted_1hour_code.to_csv(
        str(path_1hour) + "/diff_percent_from_code_-_full_segment_time_series.csv"
    )

    print("\tWritten")
