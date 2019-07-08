import numpy as np
import pandas as pd
import datetime

import feather

from tqdm import tqdm

from argparse import ArgumentParser
import os.path
from pathlib import Path


def filter_rare(stop_events):

    print("Calculating rare segments...")

    segment_counts = stop_events.groupby("segment_code").size()

    filtered_stop_events = stop_events.drop(
        stop_events[
            stop_events["segment_code"].isin(
                segment_counts[segment_counts < 120].index.values
            )
        ].index
    )

    print("\tCalculated")

    return filtered_stop_events


def add_offsets_inner(row, ts):

    try:
        if row[1] == "":
            return np.nan

        value = ts.at[row[0], row[1]]
    except KeyError:
        value = np.nan
    return value


def add_offsets(se):

    print("Adding Offsets...")

    se["arrival_5mins"] = se["actualArrival"].dt.round("5min")
    se["offset_timestamp_5_1"] = se["arrival_5mins"] - pd.Timedelta("5 min")
    se["offset_timestamp_5_2"] = se["arrival_5mins"] - pd.Timedelta("10 min")
    se["offset_timestamp_5_3"] = se["arrival_5mins"] - pd.Timedelta("15 min")
    se["offset_timestamp_5_4"] = se["arrival_5mins"] - pd.Timedelta("20 min")

    # We need to generate this from scratch as we need both test and train data.
    ts_5 = se.pivot_table(
        index="arrival_5mins",
        columns="segment_code",
        values="diff_percent_full_segment_and_median_by_segment_code_and_hour_and_day",
        aggfunc=np.median,
    )

    ts_interp_5 = ts_5.interpolate(method="time", axis=0)

    weather = pd.read_csv("weather/CDO9610867945337_weather.csv")

    weather["date"] = pd.to_datetime(weather[" YEARMODA"], format="%Y%m%d")
    weather["rain"] = weather["PRCP  "].apply(lambda x: float(x[:-1]))

    just_rain = weather[["date", "rain"]]

    se = se.merge(just_rain, left_on="date", right_on="date")

    se["diff_segment_and_median_by_segment_code"] = (
        se["segment_duration"] - se["median_durations_by_segment_code"]
    )
    se["diff_segment_and_median_by_segment_code_and_hour_and_day"] = (
        se["segment_duration"] - se["median_durations_by_segment_code_and_hour_and_day"]
    )

    se["diff_percent_segment_and_median_by_segment_code"] = (
        se["diff_segment_and_median_by_segment_code"]
        * 100
        / se["median_durations_by_segment_code"]
    )

    se["diff_percent_segment_and_median_by_segment_code_and_hour_and_day"] = (
        se["diff_segment_and_median_by_segment_code_and_hour_and_day"]
        * 100
        / se["median_durations_by_segment_code_and_hour_and_day"]
    )

    segment_names = ["0", "segment_code"]
    column_names = ["best_offset", "self_offset"]

    for i in range(1, 5):
        segment_names.append(f"prev_segment_code_{i}")
        segment_names.append(f"next_segment_code_{i}")
        column_names.append(f"prev_stop_{i}_offset")
        column_names.append(f"next_stop_{i}_offset")

    for i in range(1, 5):
        for j in range(len(segment_names)):
            print(".", end="")
            se[f"{column_names[j]}_5_{i}"] = se[
                [f"offset_timestamp_5_{i}", segment_names[j]]
            ].apply(add_offsets_inner, axis=1, args=(ts_interp_5,))

    return se


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg  # return a filename


if __name__ == "__main__":

    parser = ArgumentParser(description="add offsets features")
    parser.add_argument(
        "-i",
        dest="input_filename",
        required=True,
        help="input feather file from a previous step",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
    )

    parser.add_argument(
        "-c",
        dest="correlation_filename",
        required=True,
        help="input feather file from a previous step",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
    )

    args = parser.parse_args()

    from_path = Path(args.input_filename)

    print("Loading data...")
    # Load in the stop_events from the previous stage in the pipeline
    stop_events = feather.read_dataframe(args.input_filename)
    stop_events = stop_events.set_index("index")

    # Ensure that the segment code is using the previous
    # timing point not the current one as we use  the previous
    # dwell time.
    stop_events["segment_code"] = (
        stop_events.prev_stopCode
        + "_"
        + stop_events.stopCode
        + "_"
        + stop_events.prev_timingPoint.str[0]
    )

    correlations = feather.read_dataframe(args.correlation_filename)
    correlations = correlations.set_index(correlations.columns[0])

    stop_events = stop_events.merge(
        correlations["0"].to_frame(),
        left_on="segment_code",
        right_index=True,
        how="left",
    )

    print("\tLoaded")

    stop_events = filter_rare(stop_events)

    stop_events = add_offsets(stop_events)

    print("Writing output file...")

    stop_events = stop_events.reset_index()

    stop_events.to_feather(
        str(from_path.parent)
        + "/stop_events_with_geo_train_test_averages_prev_next_offsets.feather"
    )

    print("\tWritten")
