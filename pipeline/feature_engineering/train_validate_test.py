import numpy as np
import pandas as pd
import datetime

from tqdm import tqdm

from argparse import ArgumentParser
import os.path
from pathlib import Path


# Do 35, 55 and 75 days (out of 89) with 14 days test each time


def split_data(stop_events, days):

    # Rest the variables train and test
    stop_events[["train", "test"]] = False

    first_day = stop_events["date"].min()

    stop_events.loc[
        stop_events["date"].isin(pd.date_range(first_day, periods=days)), "train"
    ] = True

    stop_events.loc[
        stop_events["date"].isin(
            pd.date_range(first_day + pd.Timedelta(f"{days + 1} day"), periods=14)
        ),
        "test",
    ] = True

    stop_events = stop_events[stop_events["train"] | stop_events["test"]]

    return stop_events


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg  # return a filename


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Perform train test split with cross validation"
    )

    parser.add_argument(
        "-i",
        dest="input_filename",
        required=True,
        help="input csv file from a previous step",
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

    stop_events["date"] = stop_events["date"].astype("datetime64[ns]")

    print("\tLoaded")

    print("Dropping nans...")
    # Drop any rows with nan or empty sections.
    stop_events = stop_events.dropna(axis=0)
    print("\tDropped")

    # Set up what we'll need again and again.
    stop_events = stop_events.assign(train=False, test=False)
    to_write_path = Path(str(args.input_filename)).parent

    print("Splitting Data at 35 days ...")

    stop_events_35 = split_data(stop_events, 35)

    path_35days = to_write_path / Path("35days")

    path_35days.mkdir(parents=True, exist_ok=True)

    # stop_events_35.to_csv(
    #     str(path_35days) + "/stop_events_with_geo_train_test.csv", index=False
    # )

    stop_events_35 = stop_events_35.reset_index()
    stop_events_35.to_feather(
        str(path_35days) + "/stop_events_with_geo_train_test.feather"
    )

    print("Splitting Data at 55 days ...")

    stop_events_55 = split_data(stop_events, 55)

    path_55days = to_write_path / Path("55days")

    path_55days.mkdir(parents=True, exist_ok=True)

    # stop_events_55.to_csv(
    #     str(path_55days) + "/stop_events_with_geo_train_test.csv", index=False
    # )

    stop_events_55 = stop_events_55.reset_index()
    stop_events_55.to_feather(
        str(path_55days) + "/stop_events_with_geo_train_test.feather"
    )

    print("Splitting Data at 75 days ...")

    stop_events_75 = split_data(stop_events, 75)

    path_75days = to_write_path / Path("75days")

    path_75days.mkdir(parents=True, exist_ok=True)

    # stop_events_75.to_csv(
    #     str(path_75days) + "/stop_events_with_geo_train_test.csv", index=False
    # )

    stop_events_75 = stop_events_75.reset_index()
    stop_events_75.to_feather(
        str(path_75days) + "/stop_events_with_geo_train_test.feather"
    )

    print("\tSplit")
