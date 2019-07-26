import numpy as np
import pandas as pd
import datetime

import feather

from tqdm import tqdm

from argparse import ArgumentParser
import os.path
from pathlib import Path


def add_prev_next_inner(row, patterns_dict):

    stop_code = row["prev_stopCode"]
    pattern_id = row["patternId"]

    for i in range(1, 31):
        prev = f"prev_segment_code_{i}"

        if stop_code is None:
            break

        prev_stop_code = patterns_dict[pattern_id][stop_code]["prev_stop_code"]

        if prev_stop_code is None:
            break

        row[
            prev
        ] = f"{prev_stop_code}_{stop_code}_{patterns_dict[pattern_id][stop_code]['prev_stop_timing_point']}"

        stop_code = prev_stop_code

    stop_code = row["stopCode"]

    for i in range(1, 8):
        next = f"next_segment_code_{i}"

        if stop_code is None:
            break

        next_stop_code = patterns_dict[pattern_id][stop_code]["next_stop_code"]

        if next_stop_code is None:
            break

        row[
            next
        ] = f"{stop_code}_{next_stop_code}_{patterns_dict[pattern_id][stop_code]['this_stop_timing_point']}"

        stop_code = next_stop_code

    return row


def add_prev_next_all(stop_events):

    print("Loading Patterns...")

    # Load the patterns we'll need this to make sure that each bus stop is recorded in order
    patterns = pd.read_csv("Trapeze_Data/Patterns.csv")

    patterns_dict = {}

    pattern_groups = patterns.groupby("id")

    for pattern_id, pattern in pattern_groups:

        this_patterns_dict = {}

        # Make sure that they are sorted
        pattern = pattern.sort_values("sequence")

        for i in range(pattern.shape[0]):

            this_stop_code = pattern.iloc[i]["stopCode"]
            this_timing_point = pattern.iloc[i]["timingPoint"]

            this_patterns_dict[this_stop_code] = {}

            if i != 0:
                this_patterns_dict[this_stop_code]["prev_stop_code"] = pattern.iloc[
                    i - 1
                ]["stopCode"]
                this_patterns_dict[this_stop_code][
                    "prev_stop_timing_point"
                ] = pattern.iloc[i - 1]["timingPoint"]
            else:
                this_patterns_dict[this_stop_code]["prev_stop_code"] = None
                this_patterns_dict[this_stop_code]["prev_stop_timing_point"] = None

            if i + 1 != pattern.shape[0]:
                this_patterns_dict[this_stop_code]["next_stop_code"] = pattern.iloc[
                    i + 1
                ]["stopCode"]
                this_patterns_dict[this_stop_code][
                    "next_stop_timing_point"
                ] = pattern.iloc[i + 1]["timingPoint"]
            else:
                this_patterns_dict[this_stop_code]["next_stop_code"] = None
                this_patterns_dict[this_stop_code]["next_stop_timing_point"] = None

            this_patterns_dict[this_stop_code]["this_stop_code"] = this_stop_code
            this_patterns_dict[this_stop_code][
                "this_stop_timing_point"
            ] = this_timing_point

        patterns_dict[pattern_id] = this_patterns_dict

    print("\tLoaded")

    print("Adding Prev Next codes...")

    assign_dict = {}

    for i in range(1, 31):
        assign_dict[f"prev_segment_code_{i}"] = ""

    for i in range(1, 8):
        assign_dict[f"next_segment_code_{i}"] = ""

    stop_events = stop_events.assign(**assign_dict)

    stop_events_fast_lookup = (
        stop_events.reset_index()
        .set_index(["date", "workid", "segment_code"])["index"]
        .astype(int)
    )

    tqdm().pandas()

    stop_events = stop_events.progress_apply(
        add_prev_next_inner, axis=1, args=(patterns_dict,)
    )

    print("\tAdded")

    print("Adding Prev Next indices...")

    index_columns = []
    code_columns = []

    for i in range(1, 31):
        index_columns.append(f"prev_event_index_{i}")
        code_columns.append(f"prev_segment_code_{i}")

    for i in range(1, 8):
        index_columns.append(f"next_event_index_{i}")
        code_columns.append(f"next_segment_code_{i}")

    for i in tqdm(range(len(index_columns))):

        stop_events = stop_events.merge(
            stop_events_fast_lookup.rename(index_columns[i]).to_frame(),
            left_on=["date", "workid", code_columns[i]],
            right_index=True,
            how="left",
        )
        stop_events[index_columns[i]] = (
            stop_events[index_columns[i]].replace(np.nan, -1).astype(int)
        )

    print("\tAdded")

    return stop_events


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg  # return a filename


def exclude_columns_containing(se, to_remove):

    min_cols = [c for c in se.columns if not any(x in c for x in to_remove)]

    se_min = se[min_cols]

    return se_min


if __name__ == "__main__":

    parser = ArgumentParser(description="add prev next features")
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

    stop_events = exclude_columns_containing(stop_events, ["mean", "median", "offset"])

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

    print("\tLoaded")

    stop_events = add_prev_next_all(stop_events)

    print("Writing output file...")

    stop_events = stop_events.reset_index()

    stop_events.to_feather(str(from_path.parent) + "/se_prev_next.feather")

    print("\tWritten")
