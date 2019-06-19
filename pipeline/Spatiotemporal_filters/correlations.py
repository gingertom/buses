import numpy as np
import pandas as pd

print("Loading data...")
# Load in the stop_events from the previous stage in the pipeline
time_series_10mins_chd = pd.read_csv(
    "Intermediate_Data/diff_10mins_code_hour_day_percent_full_segment_time_series.csv"
)

time_series_10mins_c = pd.read_csv(
    "Intermediate_Data/diff_10mins_code_percent_full_segment_time_series.csv"
)

time_series_1hour_chd = pd.read_csv(
    "Intermediate_Data/diff_1hour_code_hour_day_percent_full_segment_time_series.csv"
)

time_series_1hour_c = pd.read_csv(
    "Intermediate_Data/diff_1hour_code_percent_full_segment_time_series.csv"
)


time_series_10mins_chd["arrival_10mins"] = time_series_10mins_chd[
    "arrival_10mins"
].astype("datetime64[ns]")

time_series_10mins_c["arrival_10mins"] = time_series_10mins_c["arrival_10mins"].astype(
    "datetime64[ns]"
)

time_series_1hour_chd["arrival_1hour"] = time_series_1hour_chd["arrival_1hour"].astype(
    "datetime64[ns]"
)

time_series_1hour_c["arrival_1hour"] = time_series_1hour_c["arrival_1hour"].astype(
    "datetime64[ns]"
)

time_series_10mins_chd.set_index("arrival_10mins")
time_series_10mins_c.set_index("arrival_10mins")

time_series_1hour_chd.set_index("arrival_1hour")
time_series_1hour_c.set_index("arrival_1hour")

print("\tLoaded")

print("Calculating correlations...")

correlations_10mins_chd = time_series_10mins_chd.corr(method="pearson", min_periods=10)
correlations_10mins_chd = correlations_10mins_chd.fillna(value=float("-inf"))

correlations_10mins_c = time_series_10mins_c.corr(method="pearson", min_periods=10)
correlations_10mins_c = correlations_10mins_c.fillna(value=float("-inf"))

correlations_1hour_chd = time_series_1hour_chd.corr(method="pearson", min_periods=10)
correlations_1hour_chd = correlations_1hour_chd.fillna(value=float("-inf"))

correlations_1hour_c = time_series_1hour_c.corr(method="pearson", min_periods=10)
correlations_1hour_c = correlations_1hour_c.fillna(value=float("-inf"))


print("\tCalculated")

print("Writing output file...")

best_correlations_10mins_chd = pd.DataFrame(
    np.fliplr(
        correlations_10mins_chd.columns[
            np.argsort(correlations_10mins_chd.values, axis=1)[:, -31:-1]
        ]
    ),
    index=correlations_10mins_chd.index,
)

best_correlations_10mins_c = pd.DataFrame(
    np.fliplr(
        correlations_10mins_c.columns[
            np.argsort(correlations_10mins_c.values, axis=1)[:, -31:-1]
        ]
    ),
    index=correlations_10mins_c.index,
)

best_correlations_1hour_chd = pd.DataFrame(
    np.fliplr(
        correlations_1hour_chd.columns[
            np.argsort(correlations_1hour_chd.values, axis=1)[:, -31:-1]
        ]
    ),
    index=correlations_1hour_chd.index,
)

best_correlations_1hour_c = pd.DataFrame(
    np.fliplr(
        correlations_1hour_c.columns[
            np.argsort(correlations_1hour_c.values, axis=1)[:, -31:-1]
        ]
    ),
    index=correlations_1hour_c.index,
)

best_correlations_10mins_chd.to_csv(
    "Intermediate_Data/best_correlations_10mins_chd.csv"
)

best_correlations_10mins_c.to_csv("Intermediate_Data/best_correlations_10mins_c.csv")

best_correlations_1hour_chd.to_csv("Intermediate_Data/best_correlations_1hour_chd.csv")

best_correlations_1hour_c.to_csv("Intermediate_Data/best_correlations_1hour_c.csv")

print("\tWritten")
