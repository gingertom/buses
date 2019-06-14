import numpy as np
import pandas as pd
import datetime

from tqdm import tqdm

from math import radians, cos, sin, asin, sqrt

CENTRE_BOURNEMOUTH = -1.88, 50.72

print("Loading data...")
# Load in the stop_events from the previous stage in the pipeline
stop_events = pd.read_csv(
    "Intermediate_Data/stop_events_with_geo_features.csv", parse_dates=[1, 5, 6, 18, 19]
)


# Force it to treat all the times as actual times.
stop_events["aimedArrival"] = stop_events["aimedArrival"].astype("datetime64[ns]")
stop_events["aimedDeparture"] = stop_events["aimedDeparture"].astype("datetime64[ns]")
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

print("\tLoaded")

print("Dropping nans...")
# Drop any rows with nan or empty sections.
stop_events = stop_events.dropna(axis=0)
print("\tDropped")

stop_events["segment_name"] = stop_events.prev_stopCode + "_" + stop_events.stopCode

print("Adding Durations...")
# Add new columns with some durations.
stop_events["dwell_duration_dest"] = (
    stop_events.actualDeparture - stop_events.actualArrival
).astype("timedelta64[s]")
stop_events["dwell_duration_prev"] = (
    stop_events.prev_actualDeparture - stop_events.prev_actualArrival
).astype("timedelta64[s]")
stop_events["segment_duration"] = (
    stop_events.actualArrival - stop_events.prev_actualDeparture
).astype("timedelta64[s]")
stop_events["timetable_segment_duration"] = (
    stop_events.aimedArrival - stop_events.prev_aimedDeparture
).astype("timedelta64[s]")
print("\tAdded")

print("Adding Means...")
# Add in columns for the day of the week and hour of the day that the bus arrives.
arrival_times = pd.to_datetime(stop_events.actualArrival)
stop_events["arrival_hour"] = arrival_times.dt.hour
stop_events["arrival_day"] = arrival_times.dt.dayofweek


# Create some new columns with the means of the durations
mean_durations_by_segment_code = (
    stop_events.groupby("segment_code")["segment_duration"]
    .mean()
    .rename("mean_durations_by_segment_code")
)
stop_events = stop_events.merge(
    mean_durations_by_segment_code.to_frame(), "left", on=["segment_code"]
)


mean_durations_by_segment_code_and_hour = (
    stop_events.groupby(["segment_code", "arrival_hour"])["segment_duration"]
    .mean()
    .rename("mean_durations_by_segment_code_and_hour")
)
stop_events = stop_events.merge(
    mean_durations_by_segment_code_and_hour.to_frame(),
    "left",
    left_on=["segment_code", "arrival_hour"],
    right_on=["segment_code", "arrival_hour"],
)

mean_durations_by_segment_code_and_hour_and_day = (
    stop_events.groupby(["segment_code", "arrival_hour", "arrival_day"])[
        "segment_duration"
    ]
    .mean()
    .rename("mean_durations_by_segment_code_and_hour_and_day")
)
stop_events = stop_events.merge(
    mean_durations_by_segment_code_and_hour_and_day.to_frame(),
    "left",
    left_on=["segment_code", "arrival_hour", "arrival_day"],
    right_index=True,
)

mean_dwell_dest_durations_by_stop_code = (
    stop_events.groupby("stopCode")["dwell_duration_dest"]
    .mean()
    .rename("mean_dwell_dest_durations_by_stop_code")
)
stop_events = stop_events.merge(
    mean_dwell_dest_durations_by_stop_code.to_frame(), "left", on=["stopCode"]
)

mean_dwell_prev_durations_by_stop_code = mean_dwell_dest_durations_by_stop_code.rename(
    "mean_dwell_prev_durations_by_stop_code"
)

stop_events = stop_events.merge(
    mean_dwell_prev_durations_by_stop_code.to_frame(),
    "left",
    left_on=["prev_stopCode"],
    right_on=["stopCode"],
)

mean_dwell_dest_by_stop_code_and_hour = (
    stop_events.groupby(["stopCode", "arrival_hour"])["dwell_duration_dest"]
    .mean()
    .rename("mean_dwell_dest_by_stop_code_and_hour")
)
stop_events = stop_events.merge(
    mean_dwell_dest_by_stop_code_and_hour.to_frame(),
    "left",
    left_on=["stopCode", "arrival_hour"],
    right_on=["stopCode", "arrival_hour"],
)

mean_dwell_prev_by_stop_code_and_hour = mean_dwell_dest_by_stop_code_and_hour.rename(
    "mean_dwell_prev_by_stop_code_and_hour"
)

stop_events = stop_events.merge(
    mean_dwell_prev_by_stop_code_and_hour.to_frame(),
    "left",
    left_on=["prev_stopCode", "arrival_hour"],
    right_on=["stopCode", "arrival_hour"],
)


mean_dwell_dest_by_stop_code_and_hour_and_day = (
    stop_events.groupby(["stopCode", "arrival_hour", "arrival_day"])[
        "dwell_duration_dest"
    ]
    .mean()
    .rename("mean_dwell_dest_by_stop_code_and_hour_and_day")
)
stop_events = stop_events.merge(
    mean_dwell_dest_by_stop_code_and_hour_and_day.to_frame(),
    "left",
    left_on=["stopCode", "arrival_hour", "arrival_day"],
    right_index=True,
)

mean_dwell_prev_by_stop_code_and_hour_and_day = mean_dwell_dest_by_stop_code_and_hour_and_day.rename(
    "mean_dwell_prev_by_stop_code_and_hour_and_day"
)

stop_events = stop_events.merge(
    mean_dwell_prev_by_stop_code_and_hour_and_day.to_frame(),
    "left",
    left_on=["prev_stopCode", "arrival_hour", "arrival_day"],
    right_index=True,
)

print("\tAdded")


print("Adding Medians...")

# Create some new columns with the means of the durations
median_durations_by_segment_code = (
    stop_events.groupby("segment_code")["segment_duration"]
    .median()
    .rename("median_durations_by_segment_code")
)
stop_events = stop_events.merge(
    median_durations_by_segment_code.to_frame(), "left", on=["segment_code"]
)


median_durations_by_segment_code_and_hour = (
    stop_events.groupby(["segment_code", "arrival_hour"])["segment_duration"]
    .median()
    .rename("median_durations_by_segment_code_and_hour")
)
stop_events = stop_events.merge(
    median_durations_by_segment_code_and_hour.to_frame(),
    "left",
    left_on=["segment_code", "arrival_hour"],
    right_on=["segment_code", "arrival_hour"],
)

median_durations_by_segment_code_and_hour_and_day = (
    stop_events.groupby(["segment_code", "arrival_hour", "arrival_day"])[
        "segment_duration"
    ]
    .median()
    .rename("median_durations_by_segment_code_and_hour_and_day")
)
stop_events = stop_events.merge(
    median_durations_by_segment_code_and_hour_and_day.to_frame(),
    "left",
    left_on=["segment_code", "arrival_hour", "arrival_day"],
    right_index=True,
)

median_dwell_dest_durations_by_stop_code = (
    stop_events.groupby("stopCode")["dwell_duration_dest"]
    .median()
    .rename("median_dwell_dest_durations_by_stop_code")
)
stop_events = stop_events.merge(
    median_dwell_dest_durations_by_stop_code.to_frame(), "left", on=["stopCode"]
)

median_dwell_prev_durations_by_stop_code = median_dwell_dest_durations_by_stop_code.rename(
    "median_dwell_prev_durations_by_stop_code"
)

stop_events = stop_events.merge(
    median_dwell_prev_durations_by_stop_code.to_frame(),
    "left",
    left_on=["prev_stopCode"],
    right_on=["stopCode"],
)

median_dwell_dest_by_stop_code_and_hour = (
    stop_events.groupby(["stopCode", "arrival_hour"])["dwell_duration_dest"]
    .median()
    .rename("median_dwell_dest_by_stop_code_and_hour")
)
stop_events = stop_events.merge(
    median_dwell_dest_by_stop_code_and_hour.to_frame(),
    "left",
    left_on=["stopCode", "arrival_hour"],
    right_on=["stopCode", "arrival_hour"],
)

median_dwell_prev_by_stop_code_and_hour = median_dwell_dest_by_stop_code_and_hour.rename(
    "median_dwell_prev_by_stop_code_and_hour"
)

stop_events = stop_events.merge(
    median_dwell_prev_by_stop_code_and_hour.to_frame(),
    "left",
    left_on=["prev_stopCode", "arrival_hour"],
    right_on=["stopCode", "arrival_hour"],
)


median_dwell_dest_by_stop_code_and_hour_and_day = (
    stop_events.groupby(["stopCode", "arrival_hour", "arrival_day"])[
        "dwell_duration_dest"
    ]
    .median()
    .rename("median_dwell_dest_by_stop_code_and_hour_and_day")
)
stop_events = stop_events.merge(
    median_dwell_dest_by_stop_code_and_hour_and_day.to_frame(),
    "left",
    left_on=["stopCode", "arrival_hour", "arrival_day"],
    right_index=True,
)

median_dwell_prev_by_stop_code_and_hour_and_day = median_dwell_dest_by_stop_code_and_hour_and_day.rename(
    "median_dwell_prev_by_stop_code_and_hour_and_day"
)

stop_events = stop_events.merge(
    median_dwell_prev_by_stop_code_and_hour_and_day.to_frame(),
    "left",
    left_on=["prev_stopCode", "arrival_hour", "arrival_day"],
    right_index=True,
)

print("\tAdded")

print("Writing output file...")

stop_events.to_csv("Intermediate_Data/stop_events_with_all_features.csv", index=False)

print("\tWritten")
