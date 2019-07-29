import numpy as np
import pandas as pd
import datetime

import feather

import json

from tqdm import tqdm

from argparse import ArgumentParser
import os.path
from pathlib import Path

from sklearn.linear_model import LinearRegression


def add_averages(se):
    print("Adding Medians and means...")

    se["dry"] = se["rain"] == 0
    se["weekend"] = se["arrival_day"] > 4

    se["weekend"] = se["weekend"].astype(bool)

    segment_code_and_hour_and_daytype_groups = se[se["train"]].groupby(
        ["segment_code", "arrival_hour", "weekend"]
    )

    median_prev_dwell_by_segment_code_and_hour_and_daytype = (
        segment_code_and_hour_and_daytype_groups["dwell_duration_prev"]
        .median()
        .rename("median_prev_dwell_by_segment_code_and_hour_and_daytype")
    )

    se = se.merge(
        median_prev_dwell_by_segment_code_and_hour_and_daytype.to_frame(),
        "left",
        left_on=["segment_code", "arrival_hour", "weekend"],
        right_index=True,
    )

    mean_prev_dwell_by_segment_code_and_hour_and_daytype = (
        segment_code_and_hour_and_daytype_groups["dwell_duration_prev"]
        .mean()
        .rename("mean_prev_dwell_by_segment_code_and_hour_and_daytype")
    )

    se = se.merge(
        mean_prev_dwell_by_segment_code_and_hour_and_daytype.to_frame(),
        "left",
        left_on=["segment_code", "arrival_hour", "weekend"],
        right_index=True,
    )

    se["prev_dwell_type_normal"] = True

    # If it's a timing point and the bus is early (or even 20 seconds late) treat it as not normal
    se.loc[
        (se["prev_timingPoint"] == 1)
        & (
            se["prev_actualArrival"] - pd.Timedelta("20 second")
            < se["prev_aimedArrival"]
        ),
        "prev_dwell_type_normal",
    ] = False

    segment_code_and_hour_and_daytype_groups_normal = se[
        (se["train"] is True) & (se["prev_dwell_type_normal"] is True)
    ].groupby(["segment_code", "arrival_hour", "weekend"])

    median_prev_dwell_by_segment_code_and_hour_and_daytype_normal = (
        segment_code_and_hour_and_daytype_groups_normal["dwell_duration_prev"]
        .median()
        .rename("median_prev_dwell_by_segment_code_and_hour_and_daytype_normal")
    )

    se = se.merge(
        median_prev_dwell_by_segment_code_and_hour_and_daytype_normal.to_frame(),
        "left",
        left_on=["segment_code", "arrival_hour", "weekend"],
        right_index=True,
    )

    mean_prev_dwell_by_segment_code_and_hour_and_daytype_normal = (
        segment_code_and_hour_and_daytype_groups_normal["dwell_duration_prev"]
        .mean()
        .rename("mean_prev_dwell_by_segment_code_and_hour_and_daytype_normal")
    )

    se = se.merge(
        mean_prev_dwell_by_segment_code_and_hour_and_daytype_normal.to_frame(),
        "left",
        left_on=["segment_code", "arrival_hour", "weekend"],
        right_index=True,
    )

    # There are only 268 nan's just get rid of them. Should probably replace them with the normal median?
    se = se.dropna(
        subset=["median_prev_dwell_by_segment_code_and_hour_and_daytype_normal"]
    )

    print("\tAdded")

    return se


def add_specials(se):

    print("Adding specials...")

    # How early the bus is in seconds
    se["how_early"] = (se["prev_aimedArrival"] - se["prev_actualArrival"]).astype(
        "<m8[s]"
    )

    merged_stops = '{"1290BOB20387": ["167_16", "167_125", "167_18", "167_128", "167_21", "167_14", "167_127", "167_116", "167_2", "167_3", "167_115", "167_6", "167_117", "167_118", "167_1", "167_17", "167_15", "167_126", "167_124", "168_1", "168_2", "168_116", "168_3", "168_118", "168_6", "168_117", "168_115", "168_17", "168_126", "168_18", "168_15", "168_16", "168_125", "168_128", "168_21", "168_124", "168_127", "168_14"],"1290BOB20403": ["167_38", "167_44", "167_42", "167_135", "167_137", "167_40", "167_133", "167_134", "167_46", "167_37", "167_51", "167_36", "167_48", "167_23", "167_22", "167_26", "167_130", "167_24", "167_25", "167_28", "167_129", "167_33", "167_132", "167_31", "167_35", "167_32", "167_34", "167_131", "167_45", "167_136", "167_43", "167_41", "167_39", "168_24", "168_25", "168_26", "168_130", "168_22", "168_129", "168_28", "168_23", "168_33", "168_132", "168_35", "168_31", "168_32", "168_34", "168_131", "168_45", "168_43", "168_41", "168_136", "168_134", "168_46", "168_135", "168_137", "168_39", "168_37", "168_51", "168_48", "168_38", "168_40", "168_42", "168_44", "168_133", "168_36", "170_23", "170_22", "170_91", "170_21", "170_25", "170_20", "170_19", "170_90", "170_30", "170_93", "170_28", "170_31", "170_29", "170_92", "170_33", "170_35", "170_36", "170_34", "170_32"],"1280BOB20047": ["167_267", "167_265", "167_264", "167_266", "168_265", "168_264", "168_266", "168_267", "170_217", "170_215", "170_214", "170_216"],"1290BOB20402": ["167_284", "168_284", "170_236"]}'

    merged_stops = json.loads(merged_stops)

    se["merged_stop_prev"] = False

    for stop, patterns in merged_stops.items():

        se.loc[
            (se["prev_stopCode"] == stop) & (se["patternId"].isin(patterns)),
            "merged_stop_prev",
        ] = True

    se["first_stop"] = False

    # This assumes that the data is already on chronological order which is
    # fairly good assumption.
    firsts = se.reset_index().groupby(["vehicle", "date"]).first()

    se.loc[firsts["index"], "first_stop"] = True

    print("\tAdded")

    return se


def add_gaps(se):

    print("Adding gaps...")

    se_prev_stopCode_groups = se.groupby(["prev_stopCode"])

    se["last_bus_gap"] = np.nan
    se["next_bus_gap"] = np.nan

    for _, stop in tqdm(se_prev_stopCode_groups):

        stop = stop.sort_values("prev_actualArrival")

        previous_bus = stop.shift(1)
        next_bus = stop.shift(-1)

        se.loc[stop.index, "last_bus_gap"] = (
            stop["prev_actualArrival"] - previous_bus["prev_actualArrival"]
        ).astype("<m8[s]")
        se.loc[stop.index, "next_bus_gap"] = (
            next_bus["prev_actualArrival"] - stop["prev_actualArrival"]
        ).astype("<m8[s]")

    se_prev_stopCode_rout_groups = se.groupby(["prev_stopCode", "publicName"])

    se["last_this_bus_gap"] = np.nan
    se["next_this_bus_gap"] = np.nan

    for _, stop in tqdm(se_prev_stopCode_rout_groups):

        stop = stop.sort_values("prev_actualArrival")

        previous_bus = stop.shift(1)
        next_bus = stop.shift(-1)

        se.loc[stop.index, "last_this_bus_gap"] = (
            stop["prev_actualArrival"] - previous_bus["prev_actualArrival"]
        ).astype("<m8[s]")
        se.loc[stop.index, "next_this_bus_gap"] = (
            next_bus["prev_actualArrival"] - stop["prev_actualArrival"]
        ).astype("<m8[s]")

    print("\tAdded")

    return se


def add_predictions(se):

    print("Adding predictions...")

    # For normal stops (including those that are tp but late) just use the median
    se["dwell_predict_rules_median"] = se[
        "median_prev_dwell_by_segment_code_and_hour_and_daytype_normal"
    ]

    # Same for means
    se["dwell_predict_rules_mean"] = se[
        "mean_prev_dwell_by_segment_code_and_hour_and_daytype_normal"
    ]

    # Train a linear regression line on the not-normal events between -20 and +100 seconds early
    reg = LinearRegression()

    fit_data = se.loc[
        (se["prev_timingPoint"] == 1)
        & (se["how_early"] < 100)
        & (se["how_early"] > -20),
        ["how_early", "dwell_duration_prev"],
    ].values

    reg.fit(np.array(fit_data[:, 0]).reshape(-1, 1), np.array(fit_data[:, 1]))

    # We now use the linear model to predict the behaviour of early timing point buses
    # Clip the earliness to 100 as actually if a bus is early than that it's dwell seems to be
    # linear(ish).
    se.loc[~se["prev_dwell_type_normal"], "dwell_predict_rules_median"] = reg.predict(
        np.clip(
            se.loc[~se["prev_dwell_type_normal"], "how_early"].values, -20, 100
        ).reshape(-1, 1)
    )

    se.loc[~se["prev_dwell_type_normal"], "dwell_predict_rules_mean"] = reg.predict(
        np.clip(
            se.loc[~se["prev_dwell_type_normal"], "how_early"].values, -20, 100
        ).reshape(-1, 1)
    )

    print("\tAdded")

    return se


def add_features(stop_events):

    stop_events = add_averages(stop_events)
    stop_events = add_specials(stop_events)
    stop_events = add_gaps(stop_events)
    stop_events = add_predictions(stop_events)

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

    # Ensure that the previous timing point is an int and not a string or a float
    stop_events["prev_timingPoint"] = (
        stop_events["prev_timingPoint"].astype(float).astype(int)
    )

    print("\tLoaded")

    stop_events = add_features(stop_events)

    print("Writing output file...")

    stop_events = stop_events.reset_index()

    stop_events.to_feather(
        str(from_path.parent) + "/" + str(from_path.stem) + "_dwell.feather"
    )

    print("\tWritten")
