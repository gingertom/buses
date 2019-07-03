import numpy as np
import pandas as pd

from argparse import ArgumentParser
import os.path
from pathlib import Path

import feather


def find_direct_correlations(time_series):

    print("Calculating direct correlations")

    # Need at least a 1 percent overlap in periods
    correlations = time_series.corr(
        method="pearson", min_periods=int(time_series.shape[0] / 100)
    )

    correlations = correlations.fillna(value=float("-1.1"))

    # We ignore the best value as it will be itself
    best_correlations_names = np.fliplr(
        correlations.columns[np.argsort(correlations.values, axis=1)[:, -31:-1]]
    )

    # We ignore the best value as it will be itself
    best_correlations_values = np.fliplr(
        np.sort(correlations.values, axis=1)[:, -31:-1]
    )

    best_correlations = pd.DataFrame(
        np.column_stack([best_correlations_names, best_correlations_values]),
        index=correlations.index,
    )

    print("\tCalculated")

    return best_correlations


def find_offset_correlations(time_series):

    print("Calculating offset correlations")

    time_wide = pd.concat([time_series, time_series.shift(1)], axis=1)

    # Need at least a 1 percent overlap in periods
    correlations = time_wide.corr(
        method="pearson", min_periods=int(time_series.shape[0] / 100)
    )
    correlations = correlations.fillna(value=float("-1.1"))

    # We only want the top right quater of the correlations as that relates to the
    # correlations between the direct and the offset values.
    correlations = correlations.iloc[: time_series.shape[1], time_series.shape[1] :]

    best_correlations_names = np.fliplr(
        correlations.columns[np.argsort(correlations.values, axis=1)[:, -30:]]
    )

    best_correlations_values = np.fliplr(np.sort(correlations.values, axis=1)[:, -30:])

    best_correlations = pd.DataFrame(
        np.column_stack([best_correlations_names, best_correlations_values]),
        index=correlations.index,
    )

    print("\tCalculated")

    return best_correlations


def find_high_traffic_correlations(time_series, min_fraction):

    print("Calculating high traffic correlations")

    # Only segment codes that have entries for at least 1/(fraction)
    # of all the time intervals.
    popular_segments = time_series.columns[
        (time_series.count() > (time_series.shape[0] / min_fraction))
    ].values

    time_wide = pd.concat([time_series, time_series[popular_segments]], axis=1)

    # Need at least a 1 percent overlap in periods
    correlations = time_wide.corr(
        method="pearson", min_periods=int(time_series.shape[0] / 100)
    )
    correlations = correlations.fillna(value=float("-1.1"))

    # We only want the top right quater of the correlations as that relates to the
    # correlations between the direct and the offset values.
    correlations = correlations.iloc[: time_series.shape[1], time_series.shape[1] :]

    best_correlations_names = np.fliplr(
        correlations.columns[np.argsort(correlations.values, axis=1)[:, -31:]]
    )

    mask = best_correlations_names[:, 0] == correlations.index

    best_correlations_names[mask] = np.roll(best_correlations_names[mask], -1, axis=1)

    best_correlations_values = np.fliplr(np.sort(correlations.values, axis=1)[:, -31:])

    best_correlations_values[mask] = np.roll(best_correlations_values[mask], -1, axis=1)

    best_correlations = pd.DataFrame(
        np.column_stack(
            [best_correlations_names[:, :30], best_correlations_values[:, :30]]
        ),
        index=correlations.index,
    )

    print("\tCalculated")

    return best_correlations


def find_high_traffic_offset_correlations(time_series, min_fraction):

    print("Calculating high traffic correlations")

    # Only segment codes that have entries for at least 1/(fraction)
    # of all the time intervals.
    popular_segments = time_series.columns[
        (time_series.count() > (time_series.shape[0] / min_fraction))
    ].values

    time_wide = pd.concat([time_series, time_series[popular_segments].shift(1)], axis=1)

    # Need at least a 1 percent overlap in periods
    correlations = time_wide.corr(
        method="pearson", min_periods=int(time_series.shape[0] / 100)
    )
    correlations = correlations.fillna(value=float("-1.1"))

    # We only want the top right quater of the correlations as that relates to the
    # correlations between the direct and the offset values.
    correlations = correlations.iloc[: time_series.shape[1], time_series.shape[1] :]

    best_correlations_names = np.fliplr(
        correlations.columns[np.argsort(correlations.values, axis=1)[:, -30:]]
    )

    best_correlations_values = np.fliplr(np.sort(correlations.values, axis=1)[:, -30:])

    best_correlations = pd.DataFrame(
        np.column_stack(
            [best_correlations_names[:, :30], best_correlations_values[:, :30]]
        ),
        index=correlations.index,
    )

    print("\tCalculated")

    return best_correlations


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg  # return filename


def write_output(
    best_correlations, offset=False, interpolated=False, high_traffic=False
):
    print("Writing output file...")

    filename = "/best_correlations_"

    if interpolated:
        filename += "interp_"

    if high_traffic:
        filename += "high_traffic_"

    if offset:
        filename += "offset_"

    filename += str(list(from_path.parts)[-2]) + "_code"

    if "code_hour_day" in str(from_path.stem):
        filename += "_hour_day"

    filename += ".feather"

    best_correlations = best_correlations.reset_index()
    best_correlations.columns = map(str, best_correlations.columns)
    best_correlations.to_feather(str(from_path.parent) + filename)

    print("\tWritten")


if __name__ == "__main__":

    parser = ArgumentParser(description="Calculate correlations")
    parser.add_argument(
        "-i",
        dest="input_filename",
        required=True,
        help="input feather file from a previous stage",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
    )

    args = parser.parse_args()

    from_path = Path(args.input_filename)

    print(f"Loading data...{str(args.input_filename)}")
    # Load in the time_series from the previous stage in the pipeline
    time_series = feather.read_dataframe(args.input_filename)
    time_series = time_series.set_index(time_series.columns[0])

    print("\tLoaded")

    interp = "interp_" in str(args.input_filename)

    best_correlations = find_direct_correlations(time_series)

    write_output(best_correlations, False, interp)

    best_correlations = find_offset_correlations(time_series)

    write_output(best_correlations, True, interp)

    best_correlations = find_high_traffic_correlations(time_series, 4)

    write_output(best_correlations, False, interp, True)

    best_correlations = find_high_traffic_offset_correlations(time_series, 4)

    write_output(best_correlations, True, interp, True)
