# Buses

Software used in MSc Data Science Dissertation Project: "Limitations of Spatiotemporal Bus Prediction in a Town with Infrequent Buses".  

## Installation

This project assumes Python 3.7 and Anaconda are already installed. 

environment.yml file contains the full lost of installed packages. 

## Usage

All code requires bus data from Trapeze Group which I am not free to share openly. 

Assuming that you have independent access to this data the following sequence is recommended. 

1. Use `pipeline/bournemouth_input/data_reader.py` To load and parse the data and convert into into stop events. 
2. Use the various scripts in `pipeline/feature_engineering/*` to add features and derive new derivative files such as time series and correlation files. Recommended order: 
    - `filter_rate_and_overtakes.py`
    - `train_validate_test.py`
    - `add_features.py`
    - `add_geo_features.py`
    - `add_prev_next.py`
    - `add_offsets.py`
    - others as needed. 
3. Explore and test the various methods using Jupyter Notebooks in the `Data Exploration` folder. Some files of note:
    - Exploratory Data Analysis
        - `IEA\Stop_events EDA2.ipynb` Early investigations of stop events, mostly simple patterns. 
        - `IEA\just segments.ipynb` Looking at segment based statistics. 
        - `IEA\Contour plots.ipynb` Looking the wider context. 
    - Spatiotemporal Models
        - `short term\Correlation Coefficients.ipynb` Looking at correlations for exogenous modelling.
        - `short term\Correlation Coefficients2.ipynb` Looking at correlations for exogenous modelling.
        - `short term\first predictons.ipynb` Using exogenous models.
        - `short term\brute force predictors.ipynb` Wrapper models.
        - `GPS-speed\GPS-speed.ipynb` Using GPS speed to estimate durations. 
4. Run `pipeline/models/Exogenous Models.py` to generate the most positive results. 

## License

Licensed for non-commercial usage only. See `Licence.md`