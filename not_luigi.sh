#!/bin/bash

# /Users/tommelamed/anaconda3/envs/busses/bin/python "pipeline/bournemouth_input/data_reader.py"

# /Users/tommelamed/anaconda3/envs/busses/bin/python "pipeline/feature_engineering/add_geo_features.py" -i "data_files/B/stop_events.csv" -o "data_files/B/unfiltered/stop_events_with_geo.csv"

/Users/tommelamed/anaconda3/envs/busses/bin/python "pipeline/feature_engineering/filter_rare_and_overtakes.py" -i "data_files/B/unfiltered/stop_events_with_geo.csv" -once "data_files/B/once/stop_events_with_geo.csv" -twice "data_files/B/twice/stop_events_with_geo.csv"

/Users/tommelamed/anaconda3/envs/busses/bin/python "pipeline/feature_engineering/add_features.py" -i "data_files/B/once/stop_events_with_geo.csv" -o "data_files/B/once/stop_events_with_geo_averages.csv" 

/Users/tommelamed/anaconda3/envs/busses/bin/python "pipeline/feature_engineering/add_features.py" -i "data_files/B/twice/stop_events_with_geo.csv" -o "data_files/B/twice/stop_events_with_geo_averages.csv" 

# /Users/tommelamed/anaconda3/envs/busses/bin/python "pipeline/feature_engineering/add_features.py" -i "data_files/B/unfiltered/stop_events_with_geo.csv" -o "data_files/B/unfiltered/stop_events_with_geo_averages.csv" 

/Users/tommelamed/anaconda3/envs/busses/bin/python "pipeline/feature_engineering/diff_percent_full_segment_time_series.py" -i "data_files/B/once/stop_events_with_geo_averages.csv" 

/Users/tommelamed/anaconda3/envs/busses/bin/python "pipeline/feature_engineering/diff_percent_full_segment_time_series.py" -i "data_files/B/twice/stop_events_with_geo_averages.csv" 

# /Users/tommelamed/anaconda3/envs/busses/bin/python "pipeline/feature_engineering/diff_percent_full_segment_time_series.py" -i "data_files/B/unfiltered/stop_events_with_geo_averages.csv" 

# Find all the time series files:
find . -name "diff_percent_from_code_-_full_segment_time_series.csv" -exec /Users/tommelamed/anaconda3/envs/busses/bin/python "pipeline/Spatiotemporal_filters/correlations.py" -i {} \;

find . -name "diff_percent_from_code_hour_day_-_full_segment_time_series.csv" -exec /Users/tommelamed/anaconda3/envs/busses/bin/python "pipeline/Spatiotemporal_filters/correlations.py" -i {} \;