#!/bin/bash

PYT=/Users/tommelamed/anaconda3/envs/busses/bin/python

# print_progress() {
#     echo "###############################"
#     echo "$1"
#     echo "###############################"
# }

# if [ ! -f "data_files/B/stop_events.csv" ]; then

#     print_progress "Starting Data Reader"
#     $PYT "pipeline/bournemouth_input/data_reader.py"
# fi

# if [ ! -f "data_files/B/unfiltered/stop_events_with_geo.csv" ]; then

#     print_progress "Starting Geo Features"

#     $PYT "pipeline/feature_engineering/add_geo_features.py" \
#         -i "data_files/B/stop_events.csv" \
#         -o "data_files/B/unfiltered/stop_events_with_geo.csv"
# fi

# if [ ! -f "data_files/B/once/stop_events_with_geo.csv" ] ; then

#     print_progress "Starting Rare and overtakes"

#     $PYT "pipeline/feature_engineering/filter_rare_and_overtakes.py" \
#         -i "data_files/B/unfiltered/stop_events_with_geo.csv" \
#         -once "data_files/B/once/stop_events_with_geo.csv" 
# fi

# if [ ! -f "data_files/B/once/35days/stop_events_with_geo_train_test.feather" ] ; then

#     print_progress "Starting Train Test Split (once)"

#     $PYT "pipeline/feature_engineering/train_validate_test.py" \
#         -i "data_files/B/once/stop_events_with_geo.csv" 
# fi

# if [ ! -f "data_files/B/unfiltered/35days/stop_events_with_geo_train_test.feather" ] ; then

#     print_progress "Starting Train Test Split (unfiltered)"

#     $PYT "pipeline/feature_engineering/train_validate_test.py" \
#         -i "data_files/B/unfiltered/stop_events_with_geo.csv" 
# fi

# # Find all the train test files:
# find . -name "stop_events_with_geo_train_test.feather" \
#     -exec $PYT \
#     "pipeline/feature_engineering/add_features.py" -i {} \
#     \;

# # Find all the averages files:
# find . -name "stop_events_with_geo_train_test_averages.feather" \
#     -exec $PYT \
#     "pipeline/feature_engineering/diff_percent_full_segment_time_series.py" -i {} \
#     \;

# Find all the time series files:
find . -name "diff_percent_from_code_-_full_segment_time_series.feather" \
    -exec $PYT \
    "pipeline/Spatiotemporal_filters/correlations.py" -i {} \;

find . -name "diff_percent_from_code_hour_day_-_full_segment_time_series.feather" \
    -exec $PYT \
    "pipeline/Spatiotemporal_filters/correlations.py" -i {} \;