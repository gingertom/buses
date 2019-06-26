import numpy as np
import pandas as pd
import datetime

from tqdm import tqdm

from argparse import ArgumentParser
import os.path
from pathlib import Path

import feather


def MAPE(forecast, actual):

    if len(forecast) != len(actual):
        raise ValueError(
            "Could not calculate MAPE, forecast and actual arrays are different length"
        )

    forecast = np.asarray(forecast)
    actual = np.asarray(actual)

    with np.errstate(divide="ignore", invalid="ignore"):

        division = (actual - forecast) / actual

        division[actual == 0] = 0

        # Instead of dividing by n we count by the number of non-zero values.
        # Essentially ignoring all cases where the actual value is zero.
        mape = 100 / np.count_nonzero(actual) * np.sum(np.abs(division))

    return mape


def print_mape(stop_events):

    mean_offset = np.mean(
        stop_events[stop_events["train"]]["full_duration"]
        - stop_events[stop_events["train"]]["timetable_segment_duration"]
    )

    print(f"Mean full duration offset {mean_offset} (train)")

    mean_full_duration = np.mean(stop_events[stop_events["train"]]["full_duration"])

    print(f"Mean full duration {mean_full_duration} (train)")

    print(
        f"Mape against mean code {MAPE(stop_events[stop_events['test']]['mean_durations_by_segment_code'] + stop_events[stop_events['test']]['mean_dwell_prev_durations_by_stop_code'], stop_events[stop_events['test']]['full_duration'])} (test)"
    )

    print(
        f"Mape against mean code and hour {MAPE(stop_events[stop_events['test']]['mean_durations_by_segment_code_and_hour'] + stop_events[stop_events['test']]['mean_dwell_prev_by_stop_code_and_hour'], stop_events[stop_events['test']]['full_duration'])} (test)"
    )

    print(
        f"Mape against mean code and hour and day {MAPE(stop_events[stop_events['test']]['mean_durations_by_segment_code_and_hour_and_day'] + stop_events[stop_events['test']]['mean_dwell_prev_by_stop_code_and_hour_and_day'], stop_events[stop_events['test']]['full_duration'])} (test)"
    )

    print(
        f"Mape against median code {MAPE(stop_events[stop_events['test']]['median_full_durations_by_segment_code'], stop_events[stop_events['test']]['full_duration'])} (test)"
    )

    print(
        f"Mape against median code and hour {MAPE(stop_events[stop_events['test']]['median_full_durations_by_segment_code_and_hour'], stop_events[stop_events['test']]['full_duration'])} (test)"
    )

    print(
        f"Mape against median code and hour and day {MAPE(stop_events[stop_events['test']]['median_full_durations_by_segment_code_and_hour_and_day'], stop_events[stop_events['test']]['full_duration'])} (test)"
    )


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg  # return a filename


def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result


if __name__ == "__main__":

    # # parser = ArgumentParser(description="Calculate Mapes")

    # # parser.add_argument(
    # #     "-i",
    # #     dest="input_filename",
    # #     required=True,
    # #     help="input csv file from a previous step",
    # #     metavar="FILE",
    # #     type=lambda x: is_valid_file(parser, x),
    # # )

    # # # parser.add_argument(
    # # #     "-o",
    # # #     dest="output_filename",
    # # #     required=True,
    # # #     help="file name and path to write to",
    # # #     metavar="FILE",
    # # # )

    # # args = parser.parse_args()

    # print(
    #     f"#####################\nLoading data from: {args.input_filename}\n#####################\n"
    # )

    files = find_all("stop_events_with_geo_train_test_averages.feather", ".")

    for file in files:

        path = Path(file)

        print(f"Loading from: {str(path.parts[2])} -> {str(path.parts[3])}")

        stop_events = feather.read_dataframe(file)
        stop_events = stop_events.set_index("index")

        stop_events = stop_events.dropna()

        print_mape(stop_events)
