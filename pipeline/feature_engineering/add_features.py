import numpy as np
import pandas as pd
import datetime

import feather

from tqdm import tqdm

from argparse import ArgumentParser
import os.path
from pathlib import Path


def add_durations(stop_events):
    print("Adding Durations...")
    # Add new columns with some durations.
    stop_events["dwell_duration_dest"] = (
        stop_events.actualDeparture - stop_events.actualArrival
    ).astype("timedelta64[s]")
    stop_events["dwell_duration_prev"] = (
        stop_events.prev_actualDeparture - stop_events.prev_actualArrival
    ).astype("timedelta64[s]")
    stop_events["segment_duration"] = (
        stop_events.actualArrival - stop_events.prev_actualDeparture
    ).astype("timedelta64[s]")
    stop_events["timetable_segment_duration"] = (
        stop_events.aimedArrival - stop_events.prev_aimedDeparture
    ).astype("timedelta64[s]")

    stop_events["full_duration"] = (
        stop_events["dwell_duration_prev"] + stop_events["segment_duration"]
    )

    print("\tAdded")

    return stop_events


def add_means_and_medians(stop_events):
    print("Adding Means...")
    # Add in columns for the day of the week and hour of the day that the bus arrives.
    arrival_times = pd.to_datetime(stop_events.actualArrival)
    stop_events["arrival_hour"] = arrival_times.dt.hour
    stop_events["arrival_day"] = arrival_times.dt.dayofweek

    # Create some new columns with the means of the durations
    # We only use the stop events with train=true when making
    # the means and medians but we then apply those values to
    # all events including test

    segment_code_groups = stop_events[stop_events["train"]].groupby("segment_code")

    mean_durations_by_segment_code = (
        segment_code_groups["segment_duration"]
        .mean()
        .rename("mean_durations_by_segment_code")
    )
    stop_events = stop_events.merge(
        mean_durations_by_segment_code.to_frame(), "left", on=["segment_code"]
    )

    segment_code_and_hour_groups = stop_events[stop_events["train"]].groupby(
        ["segment_code", "arrival_hour"]
    )

    mean_durations_by_segment_code_and_hour = (
        segment_code_and_hour_groups["segment_duration"]
        .mean()
        .rename("mean_durations_by_segment_code_and_hour")
    )
    stop_events = stop_events.merge(
        mean_durations_by_segment_code_and_hour.to_frame(),
        "left",
        left_on=["segment_code", "arrival_hour"],
        right_on=["segment_code", "arrival_hour"],
    )

    segment_code_and_hour_and_day_groups = stop_events[stop_events["train"]].groupby(
        ["segment_code", "arrival_hour", "arrival_day"]
    )

    mean_durations_by_segment_code_and_hour_and_day = (
        segment_code_and_hour_and_day_groups["segment_duration"]
        .mean()
        .rename("mean_durations_by_segment_code_and_hour_and_day")
    )
    stop_events = stop_events.merge(
        mean_durations_by_segment_code_and_hour_and_day.to_frame(),
        "left",
        left_on=["segment_code", "arrival_hour", "arrival_day"],
        right_index=True,
    )

    mean_dwell_dest_durations_by_stop_code = (
        stop_events[stop_events["train"]]
        .groupby("stopCode")["dwell_duration_dest"]
        .mean()
        .rename("mean_dwell_dest_durations_by_stop_code")
    )
    stop_events = stop_events.merge(
        mean_dwell_dest_durations_by_stop_code.to_frame(), "left", on=["stopCode"]
    )

    mean_dwell_prev_durations_by_stop_code = mean_dwell_dest_durations_by_stop_code.rename(
        "mean_dwell_prev_durations_by_stop_code"
    )

    stop_events = stop_events.merge(
        mean_dwell_prev_durations_by_stop_code.to_frame(),
        "left",
        left_on=["prev_stopCode"],
        right_on=["stopCode"],
    )

    mean_dwell_dest_by_stop_code_and_hour = (
        stop_events[stop_events["train"]]
        .groupby(["stopCode", "arrival_hour"])["dwell_duration_dest"]
        .mean()
        .rename("mean_dwell_dest_by_stop_code_and_hour")
    )
    stop_events = stop_events.merge(
        mean_dwell_dest_by_stop_code_and_hour.to_frame(),
        "left",
        left_on=["stopCode", "arrival_hour"],
        right_on=["stopCode", "arrival_hour"],
    )

    mean_dwell_prev_by_stop_code_and_hour = mean_dwell_dest_by_stop_code_and_hour.rename(
        "mean_dwell_prev_by_stop_code_and_hour"
    )

    stop_events = stop_events.merge(
        mean_dwell_prev_by_stop_code_and_hour.to_frame(),
        "left",
        left_on=["prev_stopCode", "arrival_hour"],
        right_on=["stopCode", "arrival_hour"],
    )

    mean_dwell_dest_by_stop_code_and_hour_and_day = (
        stop_events[stop_events["train"]]
        .groupby(["stopCode", "arrival_hour", "arrival_day"])["dwell_duration_dest"]
        .mean()
        .rename("mean_dwell_dest_by_stop_code_and_hour_and_day")
    )
    stop_events = stop_events.merge(
        mean_dwell_dest_by_stop_code_and_hour_and_day.to_frame(),
        "left",
        left_on=["stopCode", "arrival_hour", "arrival_day"],
        right_index=True,
    )

    mean_dwell_prev_by_stop_code_and_hour_and_day = mean_dwell_dest_by_stop_code_and_hour_and_day.rename(
        "mean_dwell_prev_by_stop_code_and_hour_and_day"
    )

    stop_events = stop_events.merge(
        mean_dwell_prev_by_stop_code_and_hour_and_day.to_frame(),
        "left",
        left_on=["prev_stopCode", "arrival_hour", "arrival_day"],
        right_index=True,
    )

    print("\tAdded")

    print("Adding Medians...")

    # Create some new columns with the means of the durations
    median_durations_by_segment_code = (
        segment_code_groups["segment_duration"]
        .median()
        .rename("median_durations_by_segment_code")
    )
    stop_events = stop_events.merge(
        median_durations_by_segment_code.to_frame(), "left", on=["segment_code"]
    )

    median_durations_by_segment_code_and_hour = (
        segment_code_and_hour_groups["segment_duration"]
        .median()
        .rename("median_durations_by_segment_code_and_hour")
    )
    stop_events = stop_events.merge(
        median_durations_by_segment_code_and_hour.to_frame(),
        "left",
        left_on=["segment_code", "arrival_hour"],
        right_on=["segment_code", "arrival_hour"],
    )

    median_durations_by_segment_code_and_hour_and_day = (
        segment_code_and_hour_and_day_groups["segment_duration"]
        .median()
        .rename("median_durations_by_segment_code_and_hour_and_day")
    )
    stop_events = stop_events.merge(
        median_durations_by_segment_code_and_hour_and_day.to_frame(),
        "left",
        left_on=["segment_code", "arrival_hour", "arrival_day"],
        right_index=True,
    )

    median_dwell_dest_durations_by_stop_code = (
        stop_events[stop_events["train"]]
        .groupby("stopCode")["dwell_duration_dest"]
        .median()
        .rename("median_dwell_dest_durations_by_stop_code")
    )
    stop_events = stop_events.merge(
        median_dwell_dest_durations_by_stop_code.to_frame(), "left", on=["stopCode"]
    )

    median_dwell_prev_durations_by_stop_code = median_dwell_dest_durations_by_stop_code.rename(
        "median_dwell_prev_durations_by_stop_code"
    )

    stop_events = stop_events.merge(
        median_dwell_prev_durations_by_stop_code.to_frame(),
        "left",
        left_on=["prev_stopCode"],
        right_on=["stopCode"],
    )

    median_dwell_dest_by_stop_code_and_hour = (
        stop_events[stop_events["train"]]
        .groupby(["stopCode", "arrival_hour"])["dwell_duration_dest"]
        .median()
        .rename("median_dwell_dest_by_stop_code_and_hour")
    )
    stop_events = stop_events.merge(
        median_dwell_dest_by_stop_code_and_hour.to_frame(),
        "left",
        left_on=["stopCode", "arrival_hour"],
        right_on=["stopCode", "arrival_hour"],
    )

    median_dwell_prev_by_stop_code_and_hour = median_dwell_dest_by_stop_code_and_hour.rename(
        "median_dwell_prev_by_stop_code_and_hour"
    )

    stop_events = stop_events.merge(
        median_dwell_prev_by_stop_code_and_hour.to_frame(),
        "left",
        left_on=["prev_stopCode", "arrival_hour"],
        right_on=["stopCode", "arrival_hour"],
    )

    median_dwell_dest_by_stop_code_and_hour_and_day = (
        stop_events[stop_events["train"]]
        .groupby(["stopCode", "arrival_hour", "arrival_day"])["dwell_duration_dest"]
        .median()
        .rename("median_dwell_dest_by_stop_code_and_hour_and_day")
    )
    stop_events = stop_events.merge(
        median_dwell_dest_by_stop_code_and_hour_and_day.to_frame(),
        "left",
        left_on=["stopCode", "arrival_hour", "arrival_day"],
        right_index=True,
    )

    median_dwell_prev_by_stop_code_and_hour_and_day = median_dwell_dest_by_stop_code_and_hour_and_day.rename(
        "median_dwell_prev_by_stop_code_and_hour_and_day"
    )

    stop_events = stop_events.merge(
        median_dwell_prev_by_stop_code_and_hour_and_day.to_frame(),
        "left",
        left_on=["prev_stopCode", "arrival_hour", "arrival_day"],
        right_index=True,
    )

    print("\tAdded")

    print("Adding full duration medians...")

    median_full_durations_by_segment_code = (
        segment_code_groups["full_duration"]
        .median()
        .rename("median_full_durations_by_segment_code")
    )
    stop_events = stop_events.merge(
        median_full_durations_by_segment_code.to_frame(), "left", on=["segment_code"]
    )

    median_full_durations_by_segment_code_and_hour = (
        segment_code_and_hour_groups["full_duration"]
        .median()
        .rename("median_full_durations_by_segment_code_and_hour")
    )
    stop_events = stop_events.merge(
        median_full_durations_by_segment_code_and_hour.to_frame(),
        "left",
        left_on=["segment_code", "arrival_hour"],
        right_index=True,
    )

    median_full_durations_by_segment_code_and_hour_and_day = (
        segment_code_and_hour_and_day_groups["full_duration"]
        .median()
        .rename("median_full_durations_by_segment_code_and_hour_and_day")
    )
    stop_events = stop_events.merge(
        median_full_durations_by_segment_code_and_hour_and_day.to_frame(),
        "left",
        left_on=["segment_code", "arrival_hour", "arrival_day"],
        right_index=True,
    )

    print("\tAdded")

    return stop_events


def add_diffs(stop_events):
    print("Adding diffs for full segment medians...")

    stop_events["diff_full_segment_and_median_by_segment_code"] = (
        stop_events["full_duration"]
        - stop_events["median_full_durations_by_segment_code"]
    )
    stop_events["diff_full_segment_and_median_by_segment_code_and_hour_and_day"] = (
        stop_events["full_duration"]
        - stop_events["median_full_durations_by_segment_code_and_hour_and_day"]
    )

    stop_events["diff_percent_full_segment_and_median_by_segment_code"] = (
        stop_events["diff_full_segment_and_median_by_segment_code"]
        * 100
        / stop_events["median_full_durations_by_segment_code"]
    )

    stop_events[
        "diff_percent_full_segment_and_median_by_segment_code_and_hour_and_day"
    ] = (
        stop_events["diff_full_segment_and_median_by_segment_code_and_hour_and_day"]
        * 100
        / stop_events["median_full_durations_by_segment_code_and_hour_and_day"]
    )

    # And now for just segments:
    stop_events["diff_segment_and_median_by_segment_code"] = (
        stop_events["segment_duration"]
        - stop_events["median_durations_by_segment_code"]
    )
    stop_events["diff_segment_and_median_by_segment_code_and_hour_and_day"] = (
        stop_events["segment_duration"]
        - stop_events["median_durations_by_segment_code_and_hour_and_day"]
    )

    stop_events["diff_percent_segment_and_median_by_segment_code"] = (
        stop_events["diff_segment_and_median_by_segment_code"]
        * 100
        / stop_events["median_durations_by_segment_code"]
    )

    stop_events["diff_percent_segment_and_median_by_segment_code_and_hour_and_day"] = (
        stop_events["diff_segment_and_median_by_segment_code_and_hour_and_day"]
        * 100
        / stop_events["median_durations_by_segment_code_and_hour_and_day"]
    )

    # Now we calculate the standard deviation of the diff percentages, so we know what
    # kind of spread to predict for this code/hour/day
    segment_code_and_hour_and_day_groups = stop_events[stop_events["train"]].groupby(
        ["segment_code", "arrival_hour", "arrival_day"]
    )

    std_diff_percent_segment_median_by_segment_code_and_hour_and_day = (
        segment_code_and_hour_and_day_groups[
            "diff_percent_segment_and_median_by_segment_code_and_hour_and_day"
        ]
        .std()
        .rename("std_diff_percent_segment_median_by_segment_code_and_hour_and_day")
    )
    stop_events = stop_events.merge(
        std_diff_percent_segment_median_by_segment_code_and_hour_and_day.to_frame(),
        "left",
        left_on=["segment_code", "arrival_hour", "arrival_day"],
        right_index=True,
    )

    print("\tAdded")

    return stop_events


def add_rain(stop_events):

    weather = pd.read_csv("weather/CDO9610867945337_weather.csv")

    weather["date"] = pd.to_datetime(weather[" YEARMODA"], format="%Y%m%d")
    weather["rain"] = weather["PRCP  "].apply(lambda x: float(x[:-1]))

    just_rain = weather[["date", "rain"]]

    stop_events = stop_events.merge(just_rain, left_on="date", right_on="date")

    return stop_events


def add_features(stop_events):

    stop_events = add_rain(stop_events)
    stop_events = add_durations(stop_events)
    stop_events = add_means_and_medians(stop_events)
    stop_events = add_diffs(stop_events)

    return stop_events


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg  # return a filename


if __name__ == "__main__":

    parser = ArgumentParser(description="add averages features")
    parser.add_argument(
        "-i",
        dest="input_filename",
        required=True,
        help="input feather file from a previous step",
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

    from_path = Path(args.input_filename)

    print("Loading data...")
    # Load in the stop_events from the previous stage in the pipeline
    stop_events = feather.read_dataframe(args.input_filename)
    stop_events = stop_events.set_index("index")

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

    # Ensure that the segment code is useing the previous
    # timing point not the current one as we use  the previous
    # dwell time.
    stop_events["segment_code"] = (
        stop_events.prev_stopCode
        + "_"
        + stop_events.stopCode
        + "_"
        + stop_events.prev_timingPoint.str[0]
    )

    print("\tLoaded")

    stop_events = add_features(stop_events)

    print("Writing output file...")

    stop_events = stop_events.reset_index()

    stop_events.to_feather(
        str(from_path.parent) + "/stop_events_with_geo_train_test_averages.feather"
    )

    print("\tWritten")
