import numpy as np
import pandas as pd
import datetime

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


# Convert aimed arrival and departure to datetime opbjects.
stop_events["aimedArrival"] = pd.to_datetime(
    stop_events["date"] + pd.to_timedelta(stop_events["aimedArrival"], unit="s")
).astype("datetime64[ns]")

stop_events["aimedDeparture"] = pd.to_datetime(
    stop_events["date"] + pd.to_timedelta(stop_events["aimedDeparture"], unit="s")
).astype("datetime64[ns]")

# Load the patterns we'll need this to make sure that each bus stop is recorded in order
patterns = pd.read_csv("Trapeze_Data/Patterns.csv")

# Group by data and id (which collectively is a unique identifier for a unit of work)
# and patternId, we don't strictly need to group by patternId but it's a handy way of getting it out
single_patterns = stop_events.groupby(["date", "id", "patternId"])

for name, group in single_patterns:
    print(name[2])

    for row_index, row in group.iterrows():
        print(row)

        # For each row see if the one above it or the one below it matches the previous stop code
        # Remember that some patterns are cyclical and so we can't just start from the top and
        # find the first stop code that matches

# # Find every row that has a valid previous stop on the same
# # pattern with the same vehicle on the same date
# stop_events["match"] = np.logical_and(
#     np.logical_and(
#         stop_events.date == stop_events.date.shift(1),
#         stop_events.vehicle == stop_events.vehicle.shift(1),
#     ),
#     stop_events.patternId == stop_events.patternId.shift(1),
# )

# # Generate new columns with the details of the previous stop, it
# # doesn't matter if this isn't valid as we'll only export the valid ones
# stop_events["prev_stopCode"] = stop_events.stopCode.shift(1)
# stop_events["prev_aimedArrival"] = stop_events.aimedArrival.shift(1)
# stop_events["prev_aimedDeparture"] = stop_events.aimedDeparture.shift(1)
# stop_events["prev_actualArrival"] = stop_events.actualArrival.shift(1)
# stop_events["prev_actualDeparture"] = stop_events.actualDeparture.shift(1)

# print(stop_events.iloc[3000:3003])

# stop_events[stop_events.match is True].to_csv("stop_events.csv", index=False)
