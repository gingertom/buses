import numpy as np
import pandas as pd
import datetime

from argparse import ArgumentParser
import os.path
from pathlib import Path

import feather


def pivot(stop_events):

    print("Pivoting data...")

    stop_events["arrival_5mins"] = stop_events["actualArrival"].dt.round("5min")
    stop_events["arrival_10mins"] = stop_events["actualArrival"].dt.round("10min")
    stop_events["arrival_1hour"] = stop_events["date"] + pd.to_timedelta(
        stop_events["arrival_hour"].values, unit="h"
    )

    pivoted_5mins_code_hour_day = stop_events[stop_events["train"]].pivot_table(
        index="arrival_5mins",
        columns="segment_code",
        values="diff_percent_segment_and_mean_by_segment_code_and_hour_and_day",
        aggfunc=np.mean,
    )

    pivoted_5mins_code = stop_events[stop_events["train"]].pivot_table(
        index="arrival_5mins",
        columns="segment_code",
        values="diff_percent_segment_and_mean_by_segment_code",
        aggfunc=np.mean,
    )

    pivoted_10mins_code_hour_day = stop_events[stop_events["train"]].pivot_table(
        index="arrival_10mins",
        columns="segment_code",
        values="diff_percent_segment_and_mean_by_segment_code_and_hour_and_day",
        aggfunc=np.mean,
    )

    pivoted_10mins_code = stop_events[stop_events["train"]].pivot_table(
        index="arrival_10mins",
        columns="segment_code",
        values="diff_percent_segment_and_mean_by_segment_code",
        aggfunc=np.mean,
    )

    pivoted_1hour_code_hour_day = stop_events[stop_events["train"]].pivot_table(
        index="arrival_1hour",
        columns="segment_code",
        values="diff_percent_segment_and_mean_by_segment_code_and_hour_and_day",
        aggfunc=np.mean,
    )

    pivoted_1hour_code = stop_events[stop_events["train"]].pivot_table(
        index="arrival_1hour",
        columns="segment_code",
        values="diff_percent_segment_and_mean_by_segment_code",
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
    stop_events = feather.read_dataframe(args.input_filename)
    stop_events = stop_events.set_index("index")

    stop_events["diff_segment_and_mean_by_segment_code_and_hour_and_day"] = (
        stop_events["segment_duration"]
        - stop_events["mean_durations_by_segment_code_and_hour_and_day"]
    )

    stop_events["diff_percent_segment_and_mean_by_segment_code_and_hour_and_day"] = (
        stop_events["diff_segment_and_mean_by_segment_code_and_hour_and_day"]
        * 100
        / stop_events["mean_durations_by_segment_code_and_hour_and_day"]
    )

    stop_events["diff_segment_and_mean_by_segment_code"] = (
        stop_events["segment_duration"] - stop_events["mean_durations_by_segment_code"]
    )

    stop_events["diff_percent_segment_and_mean_by_segment_code"] = (
        stop_events["diff_segment_and_mean_by_segment_code"]
        * 100
        / stop_events["mean_durations_by_segment_code"]
    )

    print("\tLoaded")

    pivoted_5mins_code_hour_day, pivoted_5mins_code, pivoted_10mins_code_hour_day, pivoted_10mins_code, pivoted_1hour_code_hour_day, pivoted_1hour_code = pivot(
        stop_events
    )

    print("Writing output file...")

    # Make sure the folder is there before we write the file to it.
    to_write_path = Path(str(args.input_filename)).parent

    path_5mins = to_write_path / Path("5mins")

    path_5mins.mkdir(parents=True, exist_ok=True)

    # interpolated_pivoted_5mins_code_hour_day = pivoted_5mins_code_hour_day.interpolate(
    #     method="time", axis=0
    # )
    # interpolated_pivoted_5mins_code = pivoted_5mins_code.interpolate(
    #     method="time", axis=0
    # )

    padded_pivoted_5mins_code_hour_day = pivoted_5mins_code_hour_day.interpolate(
        method="pad", axis=0
    )
    padded_pivoted_5mins_code = pivoted_5mins_code.interpolate(
        method="pad", axis=0, limit=5
    )

    pivoted_5mins_code_hour_day = pivoted_5mins_code_hour_day.reset_index()
    pivoted_5mins_code_hour_day.to_feather(
        str(path_5mins)
        + "/diff_percent_from_code_hour_day_-_segment_time_series_mean.feather"
    )

    # interpolated_pivoted_5mins_code_hour_day = (
    #     interpolated_pivoted_5mins_code_hour_day.reset_index()
    # )
    # interpolated_pivoted_5mins_code_hour_day.to_feather(
    #     str(path_5mins)
    #     + "/interp_diff_percent_from_code_hour_day_-_segment_time_series.feather"
    # )

    padded_pivoted_5mins_code_hour_day = (
        padded_pivoted_5mins_code_hour_day.reset_index()
    )
    padded_pivoted_5mins_code_hour_day.to_feather(
        str(path_5mins)
        + "/padded_diff_percent_from_code_hour_day_-_segment_time_series_mean.feather"
    )

    pivoted_5mins_code = pivoted_5mins_code.reset_index()
    pivoted_5mins_code.to_feather(
        str(path_5mins) + "/diff_percent_from_code_-_segment_time_series_mean.feather"
    )

    # interpolated_pivoted_5mins_code = interpolated_pivoted_5mins_code.reset_index()
    # interpolated_pivoted_5mins_code.to_feather(
    #     str(path_5mins)
    #     + "/interp_diff_percent_from_code_-_segment_time_series.feather"
    # )

    padded_pivoted_5mins_code = padded_pivoted_5mins_code.reset_index()
    padded_pivoted_5mins_code.to_feather(
        str(path_5mins)
        + "/padded_diff_percent_from_code_-_segment_time_series_mean.feather"
    )

    path_10mins = to_write_path / Path("10mins")

    path_10mins.mkdir(parents=True, exist_ok=True)

    # interpolated_pivoted_10mins_code_hour_day = pivoted_10mins_code_hour_day.interpolate(
    #     method="time", axis=0
    # )
    # interpolated_pivoted_10mins_code = pivoted_10mins_code.interpolate(
    #     method="time", axis=0
    # )

    padded_pivoted_10mins_code_hour_day = pivoted_10mins_code_hour_day.interpolate(
        method="pad", axis=0
    )
    padded_pivoted_10mins_code = pivoted_10mins_code.interpolate(
        method="pad", axis=0, limit=2
    )

    pivoted_10mins_code_hour_day = pivoted_10mins_code_hour_day.reset_index()
    pivoted_10mins_code_hour_day.to_feather(
        str(path_10mins)
        + "/diff_percent_from_code_hour_day_-_segment_time_series_mean.feather"
    )

    # interpolated_pivoted_10mins_code_hour_day = (
    #     interpolated_pivoted_10mins_code_hour_day.reset_index()
    # )
    # interpolated_pivoted_10mins_code_hour_day.to_feather(
    #     str(path_10mins)
    #     + "/interp_diff_percent_from_code_hour_day_-_segment_time_series.feather"
    # )

    padded_pivoted_10mins_code_hour_day = (
        padded_pivoted_10mins_code_hour_day.reset_index()
    )
    padded_pivoted_10mins_code_hour_day.to_feather(
        str(path_10mins)
        + "/padded_diff_percent_from_code_hour_day_-_segment_time_series_mean.feather"
    )

    pivoted_10mins_code = pivoted_10mins_code.reset_index()
    pivoted_10mins_code.to_feather(
        str(path_10mins) + "/diff_percent_from_code_-_segment_time_series_mean.feather"
    )

    # interpolated_pivoted_10mins_code = interpolated_pivoted_10mins_code.reset_index()
    # interpolated_pivoted_10mins_code.to_feather(
    #     str(path_10mins)
    #     + "/interp_diff_percent_from_code_-_segment_time_series.feather"
    # )

    padded_pivoted_10mins_code = padded_pivoted_10mins_code.reset_index()
    padded_pivoted_10mins_code.to_feather(
        str(path_10mins)
        + "/padded_diff_percent_from_code_-_segment_time_series_mean.feather"
    )

    path_1hour = to_write_path / Path("1hour")

    path_1hour.mkdir(parents=True, exist_ok=True)

    # interpolated_pivoted_1hour_code_hour_day = pivoted_1hour_code_hour_day.interpolate(
    #     method="time", axis=0
    # )
    # interpolated_pivoted_1hour_code = pivoted_1hour_code.interpolate(
    #     method="time", axis=0
    # )

    padded_pivoted_1hour_code_hour_day = pivoted_1hour_code_hour_day.interpolate(
        method="pad", axis=0
    )
    padded_pivoted_1hour_code = pivoted_1hour_code.interpolate(
        method="pad", axis=0, limit=1
    )

    pivoted_1hour_code_hour_day = pivoted_1hour_code_hour_day.reset_index()
    pivoted_1hour_code_hour_day.to_feather(
        str(path_1hour)
        + "/diff_percent_from_code_hour_day_-_segment_time_series_mean.feather"
    )

    pivoted_1hour_code = pivoted_1hour_code.reset_index()
    pivoted_1hour_code.to_feather(
        str(path_1hour) + "/diff_percent_from_code_-_segment_time_series_mean.feather"
    )

    # interpolated_pivoted_1hour_code_hour_day = (
    #     interpolated_pivoted_1hour_code_hour_day.reset_index()
    # )
    # interpolated_pivoted_1hour_code_hour_day.to_feather(
    #     str(path_1hour)
    #     + "/interp_diff_percent_from_code_hour_day_-_segment_time_series.feather"
    # )

    padded_pivoted_1hour_code_hour_day = (
        padded_pivoted_1hour_code_hour_day.reset_index()
    )
    padded_pivoted_1hour_code_hour_day.to_feather(
        str(path_1hour)
        + "/padded_diff_percent_from_code_hour_day_-_segment_time_series_mean.feather"
    )

    # interpolated_pivoted_1hour_code = interpolated_pivoted_1hour_code.reset_index()
    # interpolated_pivoted_1hour_code.to_feather(
    #     str(path_1hour)
    #     + "/interp_diff_percent_from_code_-_segment_time_series.feather"
    # )

    padded_pivoted_1hour_code = padded_pivoted_1hour_code.reset_index()
    padded_pivoted_1hour_code.to_feather(
        str(path_1hour)
        + "/padded_diff_percent_from_code_-_segment_time_series_mean.feather"
    )

    print("\tWritten")
