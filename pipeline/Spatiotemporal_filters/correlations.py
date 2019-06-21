import numpy as np
import pandas as pd

from argparse import ArgumentParser
import os.path
from pathlib import Path


def find_correlations(time_series, interval):

    arrival_name = "arrival_" + interval

    time_series[arrival_name] = time_series[arrival_name].astype("datetime64[ns]")

    time_series.set_index(arrival_name)

    correlations = time_series.corr(method="pearson", min_periods=10)
    correlations = correlations.fillna(value=float("-inf"))

    best_correlations = pd.DataFrame(
        np.fliplr(
            correlations.columns[np.argsort(correlations.values, axis=1)[:, -31:-1]]
        ),
        index=correlations.index,
    )

    return best_correlations


# print("Loading data...")
# # Load in the stop_events from the previous stage in the pipeline
# time_series_10mins_chd = pd.read_csv(
#     "Intermediate_Data/diff_10mins_code_hour_day_percent_full_segment_time_series.csv"
# )

# time_series_10mins_c = pd.read_csv(
#     "Intermediate_Data/diff_10mins_code_percent_full_segment_time_series.csv"
# )

# time_series_1hour_chd = pd.read_csv(
#     "Intermediate_Data/diff_1hour_code_hour_day_percent_full_segment_time_series.csv"
# )

# time_series_1hour_c = pd.read_csv(
#     "Intermediate_Data/diff_1hour_code_percent_full_segment_time_series.csv"
# )


# time_series_10mins_chd["arrival_10mins"] = time_series_10mins_chd[
#     "arrival_10mins"
# ].astype("datetime64[ns]")

# time_series_10mins_c["arrival_10mins"] = time_series_10mins_c["arrival_10mins"].astype(
#     "datetime64[ns]"
# )

# time_series_1hour_chd["arrival_1hour"] = time_series_1hour_chd["arrival_1hour"].astype(
#     "datetime64[ns]"
# )

# time_series_1hour_c["arrival_1hour"] = time_series_1hour_c["arrival_1hour"].astype(
#     "datetime64[ns]"
# )

# time_series_10mins_chd.set_index("arrival_10mins")
# time_series_10mins_c.set_index("arrival_10mins")

# time_series_1hour_chd.set_index("arrival_1hour")
# time_series_1hour_c.set_index("arrival_1hour")

# print("\tLoaded")

# print("Calculating correlations...")

# correlations_10mins_chd = time_series_10mins_chd.corr(method="pearson", min_periods=10)
# correlations_10mins_chd = correlations_10mins_chd.fillna(value=float("-inf"))

# correlations_10mins_c = time_series_10mins_c.corr(method="pearson", min_periods=10)
# correlations_10mins_c = correlations_10mins_c.fillna(value=float("-inf"))

# correlations_1hour_chd = time_series_1hour_chd.corr(method="pearson", min_periods=10)
# correlations_1hour_chd = correlations_1hour_chd.fillna(value=float("-inf"))

# correlations_1hour_c = time_series_1hour_c.corr(method="pearson", min_periods=10)
# correlations_1hour_c = correlations_1hour_c.fillna(value=float("-inf"))


# print("\tCalculated")

# print("Writing output file...")

# best_correlations_10mins_chd = pd.DataFrame(
#     np.fliplr(
#         correlations_10mins_chd.columns[
#             np.argsort(correlations_10mins_chd.values, axis=1)[:, -31:-1]
#         ]
#     ),
#     index=correlations_10mins_chd.index,
# )

# best_correlations_10mins_c = pd.DataFrame(
#     np.fliplr(
#         correlations_10mins_c.columns[
#             np.argsort(correlations_10mins_c.values, axis=1)[:, -31:-1]
#         ]
#     ),
#     index=correlations_10mins_c.index,
# )

# best_correlations_1hour_chd = pd.DataFrame(
#     np.fliplr(
#         correlations_1hour_chd.columns[
#             np.argsort(correlations_1hour_chd.values, axis=1)[:, -31:-1]
#         ]
#     ),
#     index=correlations_1hour_chd.index,
# )

# best_correlations_1hour_c = pd.DataFrame(
#     np.fliplr(
#         correlations_1hour_c.columns[
#             np.argsort(correlations_1hour_c.values, axis=1)[:, -31:-1]
#         ]
#     ),
#     index=correlations_1hour_c.index,
# )

# best_correlations_10mins_chd.to_csv(
#     "Intermediate_Data/best_correlations_10mins_chd.csv"
# )

# best_correlations_10mins_c.to_csv("Intermediate_Data/best_correlations_10mins_c.csv")

# best_correlations_1hour_chd.to_csv("Intermediate_Data/best_correlations_1hour_chd.csv")

# best_correlations_1hour_c.to_csv("Intermediate_Data/best_correlations_1hour_c.csv")

# print("\tWritten")


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg  # return filename


if __name__ == "__main__":

    parser = ArgumentParser(description="Calculate correlations")
    parser.add_argument(
        "-i",
        dest="input_filename",
        required=True,
        help="input csv file from a data_reader",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
    )

    args = parser.parse_args()

    from_path = Path(args.input_filename)

    print("Loading data...")
    # Load in the time_series from the previous stage in the pipeline
    time_series = pd.read_csv(args.input_filename)

    print("\tLoaded")

    best_correlations = find_correlations(time_series, str(list(from_path.parts)[-2]))

    print("Writing output file...")

    filename = "/best_correlations_" + str(list(from_path.parts)[-2]) + "_code"

    if "code_hour_day" in str(from_path.stem):
        filename += "_hour_day"

    filename += ".csv"

    best_correlations.to_csv(str(from_path.parent) + filename)

    print("\tWritten")
