import numpy as np
import pandas as pd
import datetime

import feather

from tqdm import tqdm

from argparse import ArgumentParser
import os.path
from pathlib import Path


def add_offsets_inner(row, ts):

    try:
        if row[1] == "":
            return np.nan

        value = ts.at[row[0], row[1]]
        # value = ts[row[1]].get(row[0], np.nan)
    except KeyError:
        value = np.nan
    return value


def add_offsets(se, interpolate=True):

    print("Adding Offsets...")

    se["arrival_5mins"] = se["actualArrival"].dt.round("5min")
    se["offset_timestamp_5_1"] = se["arrival_5mins"] - pd.Timedelta("5 min")
    se["offset_timestamp_5_2"] = se["arrival_5mins"] - pd.Timedelta("10 min")
    se["offset_timestamp_5_3"] = se["arrival_5mins"] - pd.Timedelta("15 min")
    se["offset_timestamp_5_4"] = se["arrival_5mins"] - pd.Timedelta("20 min")
    se["offset_timestamp_5_5"] = se["arrival_5mins"] - pd.Timedelta("25 min")
    se["offset_timestamp_5_6"] = se["arrival_5mins"] - pd.Timedelta("30 min")
    se["offset_timestamp_5_7"] = se["arrival_5mins"] - pd.Timedelta("35 min")

    # We need to generate this from scratch as we need both test and train data.
    ts_5 = se.pivot_table(
        index="arrival_5mins",
        columns="segment_code",
        values="diff_percent_segment_and_median_by_segment_code_and_hour_and_day",
        aggfunc=np.median,
    )

    if interpolate:
        ts_5 = ts_5.interpolate(method="pad", axis=0)

    segment_names = ["segment_code"]
    column_names = ["self_offset"]

    for i in range(1, 8):
        segment_names.append(f"prev_segment_code_{i}")
        segment_names.append(f"next_segment_code_{i}")
        column_names.append(f"prev_stop_{i}_offset")
        column_names.append(f"next_stop_{i}_offset")

    for i in range(1, 8):
        for j in range(len(segment_names)):
            print(".", end="", flush=True)
            se[f"{column_names[j]}_5_{i}"] = se[
                [f"offset_timestamp_5_{i}", segment_names[j]]
            ].apply(add_offsets_inner, axis=1, args=(ts_5,))

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
        "-int",
        dest="input_interpolate",
        required=True,
        help="input Should we interpolate",
    )

    # parser.add_argument(
    #     "-c",
    #     dest="correlation_filename",
    #     required=True,
    #     help="input feather file from a previous step",
    #     metavar="FILE",
    #     type=lambda x: is_valid_file(parser, x),
    # )

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

    # correlations = feather.read_dataframe(args.correlation_filename)
    # correlations = correlations.set_index(correlations.columns[0])

    # stop_events = stop_events.merge(
    #     correlations["0"].to_frame(),
    #     left_on="segment_code",
    #     right_index=True,
    #     how="left",
    # )

    print("\tLoaded")

    if args.input_interpolate == "interpolate":
        interpolate = True
    else:
        interpolate = False

    stop_events = add_offsets(stop_events, interpolate)

    print("Writing output file...")

    stop_events = stop_events.reset_index()

    if interpolate:
        stop_events.to_feather(
            str(from_path.parent)
            + "/stop_events_with_geo_train_test_averages_prev_next_offsets_interp.feather"
        )
    else:
        stop_events.to_feather(
            str(from_path.parent)
            + "/stop_events_with_geo_train_test_averages_prev_next_offsets_noninterp.feather"
        )

    print("\tWritten")
