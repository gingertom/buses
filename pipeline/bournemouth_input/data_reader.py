import numpy as np
import pandas as pd
import datetime


def copy_to_prev(df, to_row_index, from_row_index):

    df.at[to_row_index, "match"] = True

    df.at[
        to_row_index,
        [
            "prev_stopCode",
            "prev_aimedArrival",
            "prev_aimedDeparture",
            "prev_actualArrival",
            "prev_actualDeparture",
        ],
    ] = df.loc[
        from_row_index,
        [
            "stopCode",
            "aimedArrival",
            "aimedDeparture",
            "actualArrival",
            "actualDeparture",
        ],
    ].values


# Load the vehicle events which are almost what we need for stop events
vehicle_events = pd.read_csv(
    "Trapeze_Data/VehicleEvents.csv", nrows=50000, parse_dates=[1, 5, 6]
)

# Load the performed work, which tells us about patterns, vehicles etc,
# this is important for working out consecutive runs
performed_work = pd.read_csv("Trapeze_Data/PerformedWork.csv", parse_dates=[1])

# Merge the two tables
stop_events = pd.merge(
    vehicle_events,
    performed_work,
    how="left",
    left_on=["id", "date"],
    right_on=["workid", "date"],
)

# Convert aimed arrival and departure to datetime objects.
stop_events["aimedArrival"] = pd.to_datetime(
    stop_events["date"] + pd.to_timedelta(stop_events["aimedArrival"], unit="s")
).astype("datetime64[ns]")

stop_events["aimedDeparture"] = pd.to_datetime(
    stop_events["date"] + pd.to_timedelta(stop_events["aimedDeparture"], unit="s")
).astype("datetime64[ns]")

# Load the patterns we'll need this to make sure that each bus stop is recorded in order
patterns = pd.read_csv("Trapeze_Data/Patterns.csv")

# Group by data and id (which collectively is a unique identifier for a unit of work)
single_patterns = stop_events.groupby(["date", "id"])

to_write = None

for name, group in single_patterns:

    group = group.assign(
        prev_stopCode="",
        prev_aimedArrival=datetime.datetime.now(),
        prev_aimedDeparture=datetime.datetime.now(),
        prev_actualArrival=datetime.datetime.now(),
        prev_actualDeparture=datetime.datetime.now(),
        match=False,
        timingPoint=0,
    )

    # Get the list of stop codes for this pattern as a python list
    # print(group.iloc[0]["patternId"])
    stop_codes = list(
        patterns.loc[patterns.id == group.iloc[0]["patternId"]].stopCode.values
    )

    timing_points = list(
        patterns.loc[patterns.id == group.iloc[0]["patternId"]].timingPoint.values
    )

    value_counts = group["stopCode"].value_counts()

    prev_stop_code = ""
    prev_index_into_group = -1

    for stop_index, stop_code in enumerate(stop_codes):

        found_stops = group[group.stopCode == stop_code]

        # If this stop isn't recorded in the vehicle events skip it
        if found_stops.size == 0:
            continue

        index_into_group = found_stops.index[0]

        # This is the first stop of the pattern, there is no previous so we don't have a match
        if stop_index == 0 or prev_index_into_group == -1:

            prev_stop_code = stop_code
            prev_index_into_group = index_into_group
            continue

        # If this is a duplicate stop then we need to treat it specially and merge the rows
        if value_counts.loc[stop_code] > 1:

            # This is the second instance, we've already dealt with it.
            # We don't mark this as a match and we don't update
            # the previous_index_into_group as this row is ignored
            if prev_stop_code == stop_code:
                continue
            else:
                # This is the first instance of the duplicate stop, we need to put data from both bits of the stop
                # Taking advantage of the fact that we know each duplicate stop is exactly two, next to each other
                # and that the second one is always a timing point
                second_index = group[group.stopCode == stop_code].index[1]

                group.at[
                    index_into_group, ["aimedDeparture", "actualDeparture"]
                ] = group.loc[
                    second_index, ["aimedDeparture", "actualDeparture"]
                ].values

                group.at[index_into_group, "timingPoint"] = timing_points[
                    stop_index + 1
                ]
        else:
            # If it's not a duplicate then we copy over the timing point info directly
            group.at[index_into_group, "timingPoint"] = timing_points[stop_index]

        # Now we copy over the details from the last stop, missing stops
        # and duplicate stops have already been handled
        copy_to_prev(group, index_into_group, prev_index_into_group)

        prev_stop_code = stop_code
        prev_index_into_group = index_into_group

    if to_write is None:
        to_write = group
    else:
        to_write = to_write.append(group)

    print(".", end="")

print(to_write)

to_write.to_csv("stop_events.csv", index=False)
