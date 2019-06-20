import numpy as np
import pandas as pd
import datetime

from tqdm import tqdm

from math import radians, cos, sin, asin, sqrt

CENTRE_BOURNEMOUTH = -1.88, 50.72

print("Loading data...")
# Load in the stop_events from the previous stage in the pipeline
stop_events = pd.read_csv(
    "Intermediate_Data/stop_events_with_all_features.csv", parse_dates=[1, 5, 6, 18, 19]
)

# Force it to treat all the times as actual times.
stop_events["aimedArrival"] = stop_events["aimedArrival"].astype("datetime64[ns]")
stop_events["aimedDeparture"] = stop_events["aimedDeparture"].astype("datetime64[ns]")
stop_events["actualArrival"] = stop_events["actualArrival"].astype("datetime64[ns]")
stop_events["actualDeparture"] = stop_events["actualDeparture"].astype("datetime64[ns]")

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

stop_events["date"] = stop_events["date"].astype("datetime64[ns]")

print("\tLoaded")

print("Pivoting data...")

stop_events["arrival_5mins"] = stop_events["actualArrival"].dt.round("5min")
stop_events["arrival_10mins"] = stop_events["actualArrival"].dt.round("10min")
stop_events["arrival_1hour"] = stop_events["date"] + pd.to_timedelta(
    stop_events["arrival_hour"].values, unit="h"
)

pivoted_5mins_code_hour_day = stop_events.pivot_table(
    index="arrival_5mins",
    columns="segment_code",
    values="diff_percent_full_segment_and_median_by_segment_code_and_hour_and_day",
    aggfunc=np.mean,
)

pivoted_5mins_code = stop_events.pivot_table(
    index="arrival_5mins",
    columns="segment_code",
    values="diff_percent_full_segment_and_median_by_segment_code",
    aggfunc=np.mean,
)

pivoted_10mins_code_hour_day = stop_events.pivot_table(
    index="arrival_10mins",
    columns="segment_code",
    values="diff_percent_full_segment_and_median_by_segment_code_and_hour_and_day",
    aggfunc=np.mean,
)

pivoted_10mins_code = stop_events.pivot_table(
    index="arrival_10mins",
    columns="segment_code",
    values="diff_percent_full_segment_and_median_by_segment_code",
    aggfunc=np.mean,
)

pivoted_1hour_code_hour_day = stop_events.pivot_table(
    index="arrival_1hour",
    columns="segment_code",
    values="diff_percent_full_segment_and_median_by_segment_code_and_hour_and_day",
    aggfunc=np.mean,
)

pivoted_1hour_code = stop_events.pivot_table(
    index="arrival_1hour",
    columns="segment_code",
    values="diff_percent_full_segment_and_median_by_segment_code",
    aggfunc=np.mean,
)


print("\tpivoted")

print("Writing output file...")

pivoted_5mins_code_hour_day.to_csv(
    "Intermediate_Data/diff_5mins_code_hour_day_percent_full_segment_time_series.csv"
)

pivoted_5mins_code.to_csv(
    "Intermediate_Data/diff_5mins_code_percent_full_segment_time_series.csv"
)

pivoted_10mins_code_hour_day.to_csv(
    "Intermediate_Data/diff_10mins_code_hour_day_percent_full_segment_time_series.csv"
)

pivoted_10mins_code.to_csv(
    "Intermediate_Data/diff_10mins_code_percent_full_segment_time_series.csv"
)

pivoted_1hour_code_hour_day.to_csv(
    "Intermediate_Data/diff_1hour_code_hour_day_percent_full_segment_time_series.csv"
)

pivoted_1hour_code.to_csv(
    "Intermediate_Data/diff_1hour_code_percent_full_segment_time_series.csv"
)

print("\tWritten")
