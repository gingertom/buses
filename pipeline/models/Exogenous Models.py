import numpy as np
import numpy.ma as ma
import pandas as pd
import datetime
import feather

# from sklearn.linear_model import LinearRegression

# from sklearn.metrics import mean_squared_error, mean_absolute_error

import os

# import xgboost as xgb

# import plaidml.keras

# plaidml.keras.install_backend()

import keras
from keras.preprocessing import sequence
from keras import layers, Input, Model
from keras.models import Sequential
from keras.layers import (
    Flatten,
    Dense,
    Lambda,
    LSTM,
    Dropout,
    BatchNormalization,
    ConvLSTM2D,
)

from sklearn import preprocessing

# from sklearn.ensemble import RandomForestRegressor

from argparse import ArgumentParser
import os.path
from pathlib import Path

import sys

sys.path.append("pipeline/utils/")

# from pipeline.utils.stats import Stats
from stats import Stats  # noqa: E402

USE_10 = True
USE_5 = False
USE_CONV = False
BATCH_SIZE = 512


# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def filter_rare(stop_events):

    segment_counts = stop_events.groupby("segment_code").size()

    filtered_stop_events = stop_events.drop(
        stop_events[
            stop_events["segment_code"].isin(
                segment_counts[segment_counts < 120].index.values
            )
        ].index
    )

    return filtered_stop_events


def tidy_up(se):

    se = pd.get_dummies(se, columns=["arrival_hour", "arrival_day"])

    se = se.dropna(
        subset=["diff_percent_segment_and_median_by_segment_code_and_hour_and_day"]
    )

    se["clock_direction_degrees"] = se["clock_direction_degrees"].replace(
        np.nan, np.mean(se["clock_direction_degrees"])
    )

    se = prep_spatiotemporal(se)

    se = se.replace(np.nan, 0)

    to_remove = ["mean", "median", "prev_stop_", "next_stop_", "road"]

    min_cols = [c for c in se.columns if not any(x in c for x in to_remove)]

    se_min = se[min_cols].copy()

    se_min["diff_percent_segment_and_mean_by_segment_code_and_hour_and_day"] = se[
        "diff_percent_segment_and_mean_by_segment_code_and_hour_and_day"
    ]
    se_min["mean_durations_by_segment_code_and_hour_and_day"] = se[
        "mean_durations_by_segment_code_and_hour_and_day"
    ]
    se_min["std_diff_percent_segment_mean_by_segment_code_and_hour_and_day"] = se[
        "std_diff_percent_segment_mean_by_segment_code_and_hour_and_day"
    ]

    se_min["median_durations_by_segment_code_and_hour_and_day"] = se[
        "median_durations_by_segment_code_and_hour_and_day"
    ]

    se_min["test"] = se["test"]

    se_min["train"] = se["train"]

    return se, se_min


def split_train_test(events, days):

    first_day_test = events.loc[events["test"], "date"].min()

    train_days = 0
    count = 0

    # This is a little hack to add in days to account for gaps in the records.
    while train_days != days and count != 10:

        train = events.loc[
            events["date"].isin(
                pd.date_range(
                    start=(first_day_test - pd.Timedelta(f"{days + 1 + count} days")),
                    periods=(days + 1 + count),
                )
            )
        ]

        train_days = len(train.groupby("date").first())
        count += 1

    test = events[events["test"]]

    print(f"Days in training: {(train['date'].max() - train['date'].min()).days}")
    print(f"Days in Testing: {(test['date'].max() - test['date'].min()).days}")

    return train, test


def prep_matrices(se, se_min, days):

    train, test = split_train_test(se_min, days)

    train_st, test_st = split_train_test(
        se[
            [
                "road_time_series",
                "date",
                "diff_percent_segment_and_mean_by_segment_code_and_hour_and_day",
                "segment_duration",
                "train",
                "test",
            ]
        ],
        days,
    )

    # train_matrix = stop_events[stop_events['train']][['line_distance', 'to_centre_dist', 'direction_degrees', 'best_0', 'best_1', 'best_2', 'best_3', 'best_4', 'best_5', 'best_6', 'best_7', 'best_8', 'best_9', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']].values
    # train_matrix = train.drop(
    #     [
    #         "diff_percent_segment_and_mean_by_segment_code_and_hour_and_day",
    #         "date",
    #         "segment_duration",
    #     ],
    #     axis=1,
    # )
    train_matrix_st = train_st.drop(
        [
            "diff_percent_segment_and_mean_by_segment_code_and_hour_and_day",
            "date",
            "segment_duration",
            "train",
            "test",
        ],
        axis=1,
    )
    train_target = train[
        "diff_percent_segment_and_mean_by_segment_code_and_hour_and_day"
    ]

    # test_matrix = stop_events[stop_events['test']][['line_distance', 'to_centre_dist', 'direction_degrees', 'best_0', 'best_1', 'best_2', 'best_3', 'best_4', 'best_5', 'best_6', 'best_7', 'best_8', 'best_9', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']].values
    # test_matrix = test.drop(
    #     [
    #         "diff_percent_segment_and_mean_by_segment_code_and_hour_and_day",
    #         "date",
    #         "segment_duration",
    #     ],
    #     axis=1,
    # )
    test_matrix_st = test_st.drop(
        [
            "diff_percent_segment_and_mean_by_segment_code_and_hour_and_day",
            "date",
            "segment_duration",
            "train",
            "test",
        ],
        axis=1,
    )
    test_target = test["diff_percent_segment_and_mean_by_segment_code_and_hour_and_day"]

    train_matrix = train[
        [
            "line_distance",
            "to_centre_dist",
            "direction_degrees",
            "rain",
            "timetable_segment_duration",
            "std_diff_percent_segment_mean_by_segment_code_and_hour_and_day",
            "clock_direction_degrees",
            "dry",
            "weekend",
            "arrival_hour_0",
            "arrival_hour_5",
            "arrival_hour_6",
            "arrival_hour_7",
            "arrival_hour_8",
            "arrival_hour_9",
            "arrival_hour_10",
            "arrival_hour_11",
            "arrival_hour_12",
            "arrival_hour_13",
            "arrival_hour_14",
            "arrival_hour_15",
            "arrival_hour_16",
            "arrival_hour_17",
            "arrival_hour_18",
            "arrival_hour_19",
            "arrival_hour_20",
            "arrival_hour_21",
            "arrival_hour_22",
            "arrival_hour_23",
            "arrival_day_0",
            "arrival_day_1",
            "arrival_day_2",
            "arrival_day_3",
            "arrival_day_4",
            "arrival_day_5",
            "arrival_day_6",
        ]
    ]

    test_matrix = test[
        [
            "line_distance",
            "to_centre_dist",
            "direction_degrees",
            "rain",
            "timetable_segment_duration",
            "std_diff_percent_segment_mean_by_segment_code_and_hour_and_day",
            "clock_direction_degrees",
            "dry",
            "weekend",
            "arrival_hour_0",
            "arrival_hour_5",
            "arrival_hour_6",
            "arrival_hour_7",
            "arrival_hour_8",
            "arrival_hour_9",
            "arrival_hour_10",
            "arrival_hour_11",
            "arrival_hour_12",
            "arrival_hour_13",
            "arrival_hour_14",
            "arrival_hour_15",
            "arrival_hour_16",
            "arrival_hour_17",
            "arrival_hour_18",
            "arrival_hour_19",
            "arrival_hour_20",
            "arrival_hour_21",
            "arrival_hour_22",
            "arrival_hour_23",
            "arrival_day_0",
            "arrival_day_1",
            "arrival_day_2",
            "arrival_day_3",
            "arrival_day_4",
            "arrival_day_5",
            "arrival_day_6",
        ]
    ]

    train_target = np.nan_to_num(train_target)
    test_target = np.nan_to_num(test_target)

    train_target = np.clip(train_target, -1000, 10000)
    test_target = np.clip(test_target, -1000, 10000)

    train_matrix_st = np.array(train_matrix_st.values.tolist()).squeeze()
    test_matrix_st = np.array(test_matrix_st.values.tolist()).squeeze()

    scaler_matrix = preprocessing.StandardScaler().fit(train_matrix)

    train_matrix_scaled = scaler_matrix.transform(train_matrix)
    test_matrix_scaled = scaler_matrix.transform(test_matrix)

    scaler_target = preprocessing.StandardScaler().fit(train_target[:, None])

    train_target_scaled = scaler_target.transform(train_target[:, None])
    test_target_scaled = scaler_target.transform(test_target[:, None])

    st_scaler = preprocessing.StandardScaler()
    train_matrix_st_shape = train_matrix_st.shape
    test_matrix_st_shape = test_matrix_st.shape
    train_matrix_st = st_scaler.fit_transform(
        train_matrix_st.reshape(len(train_target_scaled), -1)
    ).reshape(train_matrix_st_shape)

    test_matrix_st = st_scaler.transform(
        test_matrix_st.reshape(len(test_target_scaled), -1)
    ).reshape(test_matrix_st_shape)

    return (
        train_matrix,
        train_target,
        test_matrix,
        test_target,
        train_matrix_scaled,
        train_target_scaled,
        test_matrix_scaled,
        test_target_scaled,
        train_matrix_st,
        test_matrix_st,
        scaler_target,
        test["mean_durations_by_segment_code_and_hour_and_day"].values,
        test["segment_duration"].values,
        test,
    )


def prep_spatiotemporal(se):

    if USE_5:
        se["self_offset_5_4"] = se["self_offset_5_4"].fillna(0)

        for i in range(5, 8):
            se[f"self_offset_5_{i}"] = se[f"self_offset_5_{i}"].fillna(
                se[f"self_offset_5_{i-1}"]
            )

        for i in range(1, 5):
            for j in range(4, 8):

                if i == 1:
                    se[f"prev_stop_1_offset_5_{j}"] = se[
                        f"prev_stop_1_offset_5_{j}"
                    ].fillna(se[f"self_offset_5_{j}"])

                    se[f"next_stop_1_offset_5_{j}"] = se[
                        f"next_stop_1_offset_5_{j}"
                    ].fillna(se[f"self_offset_5_{j}"])
                else:
                    se[f"prev_stop_{i}_offset_5_{j}"] = se[
                        f"prev_stop_{i}_offset_5_{j}"
                    ].fillna(se[f"prev_stop_{i-1}_offset_5_{j}"])

                    se[f"next_stop_{i}_offset_5_{j}"] = se[
                        f"prev_stop_{i}_offset_5_{j}"
                    ].fillna(se[f"prev_stop_{i-1}_offset_5_{j}"])

        for i in range(1, 5):
            j = i + 3
            se[f"road_offset_5_{i}"] = se[
                [
                    f"prev_stop_4_offset_5_{j}",
                    f"prev_stop_3_offset_5_{j}",
                    f"prev_stop_2_offset_5_{j}",
                    f"prev_stop_1_offset_5_{j}",
                    f"self_offset_5_{j}",
                    f"next_stop_1_offset_5_{j}",
                    f"next_stop_2_offset_5_{j}",
                    f"next_stop_3_offset_5_{j}",
                    f"next_stop_4_offset_5_{j}",
                ]
            ].values.tolist()

        se["road_time_series"] = se[
            ["road_offset_5_1", "road_offset_5_2", "road_offset_5_3", "road_offset_5_4"]
        ].values.tolist()

    if USE_10:

        se["self_offset_10_3"] = se["self_offset_10_3"].fillna(0)

        for i in range(4, 15):
            se[f"self_offset_10_{i}"] = se[f"self_offset_10_{i}"].fillna(
                se[f"self_offset_10_{i-1}"]
            )

        for i in range(1, 8):
            for j in range(3, 15):

                if i == 1:
                    se[f"prev_stop_1_offset_10_{j}"] = se[
                        f"prev_stop_1_offset_10_{j}"
                    ].fillna(se[f"self_offset_10_{j}"])

                    se[f"next_stop_1_offset_10_{j}"] = se[
                        f"next_stop_1_offset_10_{j}"
                    ].fillna(se[f"self_offset_10_{j}"])
                else:
                    se[f"prev_stop_{i}_offset_10_{j}"] = se[
                        f"prev_stop_{i}_offset_10_{j}"
                    ].fillna(se[f"prev_stop_{i-1}_offset_10_{j}"])

                    se[f"next_stop_{i}_offset_10_{j}"] = se[
                        f"prev_stop_{i}_offset_10_{j}"
                    ].fillna(se[f"prev_stop_{i-1}_offset_10_{j}"])

        for i in range(1, 12):
            j = i + 2
            se[f"road_offset_10_{i}"] = se[
                [
                    f"prev_stop_6_offset_10_{j}",
                    f"prev_stop_5_offset_10_{j}",
                    f"prev_stop_4_offset_10_{j}",
                    f"prev_stop_3_offset_10_{j}",
                    f"prev_stop_2_offset_10_{j}",
                    f"prev_stop_1_offset_10_{j}",
                    f"self_offset_10_{j}",
                    f"next_stop_1_offset_10_{j}",
                    f"next_stop_2_offset_10_{j}",
                    f"next_stop_3_offset_10_{j}",
                    f"next_stop_4_offset_10_{j}",
                    f"next_stop_5_offset_10_{j}",
                    f"next_stop_6_offset_10_{j}",
                ]
            ].values.tolist()

        se["road_time_series"] = se[
            [
                "road_offset_10_1",
                "road_offset_10_2",
                "road_offset_10_3",
                "road_offset_10_4",
                "road_offset_10_5",
                "road_offset_10_6",
                "road_offset_10_7",
                "road_offset_10_8",
                "road_offset_10_9",
                "road_offset_10_10",
                "road_offset_10_11",
            ]
        ].values.tolist()

    return se


def sort_data(se, days):

    if "test" in se.columns:
        print("test in se at start of sort_data")

    se = filter_rare(se)

    # se = drop_features(se)

    se, se_min = tidy_up(se)

    return prep_matrices(se, se_min, days)


def create_fully_connected_small(input_width, dropout):
    model = Sequential()

    model.add(Dense(32, input_shape=(input_width,), activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    # model.add(layers.Dropout(rate=dropout))
    model.add(Dense(1, activation="tanh"))
    model.summary()

    return model


def create_fully_connected_medium(input_width, dropout):
    model = Sequential()

    model.add(Dense(64, input_shape=(input_width,), activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(32, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    # model.add(layers.Dropout(rate=dropout))
    model.add(Dense(1, activation="tanh"))
    model.summary()

    return model


def create_fully_connected_large(input_width, dropout):
    model = Sequential()

    model.add(Dense(128, input_shape=(input_width,), activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(62, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(32, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    # model.add(layers.Dropout(rate=dropout))
    model.add(Dense(1, activation="tanh"))
    model.summary()

    return model


def create_LSTM_small(input_shape, dropout, recurant_dropout):

    model = Sequential()

    # New Start
    model.add(LSTM(40, input_shape=input_shape, recurrent_dropout=recurant_dropout))

    # Old start
    # model.add(LSTM(40, input_shape=input_shape))

    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(32, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    # model.add(layers.Dropout(rate=dropout))
    model.add(Dense(1, activation="tanh"))
    model.summary()

    return model


def create_LSTM_medium(input_shape, dropout, recurant_dropout):

    model = Sequential()

    # New Start
    model.add(LSTM(60, input_shape=input_shape, recurrent_dropout=recurant_dropout))

    # Old start
    # model.add(LSTM(40, input_shape=input_shape))

    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(64, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(32, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    # model.add(layers.Dropout(rate=dropout))
    model.add(Dense(1, activation="tanh"))
    model.summary()

    return model


def create_LSTM_large(input_shape, dropout, recurant_dropout):

    model = Sequential()

    # New Start
    model.add(
        LSTM(
            40,
            input_shape=input_shape,
            recurrent_dropout=recurant_dropout,
            return_sequences=True,
        )
    )
    model.add(LSTM(40, recurrent_dropout=recurant_dropout))

    # Old start
    # model.add(LSTM(40, input_shape=input_shape))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(128, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(64, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(32, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    # model.add(layers.Dropout(rate=dropout))
    model.add(Dense(1, activation="tanh"))
    model.summary()

    return model


def create_LSTM_conv(input_shape, dropout, recurant_dropout):

    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_shape[0], input_shape[1], 1, 1)))
    model.add(
        ConvLSTM2D(
            filters=32, kernel_size=(5, 1), padding="same", return_sequences=True
        )
    )

    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(
        ConvLSTM2D(
            filters=32, kernel_size=(5, 1), padding="same", return_sequences=False
        )
    )

    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(64, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(32, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    model.add(layers.Dropout(rate=dropout))
    model.add(Dense(12, activation="relu"))
    # model.add(layers.Dropout(rate=dropout))
    model.add(Dense(1, activation="tanh"))
    model.summary()

    return model


def create_combined_model(
    road_input_shape, aux_input_shape, dropout, recurant_dropout, size
):

    # with help from: https://keras.io/getting-started/functional-api-guide/

    # Headline input: meant to receive road time series.
    main_input = Input(shape=road_input_shape, dtype="float32", name="road_time_input")

    if size == "small":
        lstm_out = LSTM(40, recurrent_dropout=recurant_dropout)(main_input)

    if size == "medium":
        lstm_out = LSTM(60, recurrent_dropout=recurant_dropout)(main_input)

    if size == "large":
        lstm_out = LSTM(40, recurrent_dropout=recurant_dropout, return_sequences=True)(
            main_input
        )
        lstm_out = LSTM(40, recurrent_dropout=recurant_dropout)(lstm_out)

    auxiliary_output = Dense(1, activation="tanh", name="aux_output")(lstm_out)

    auxiliary_input = Input(shape=(aux_input_shape,), name="aux_input")
    x = keras.layers.concatenate([lstm_out, auxiliary_input])

    # We stack a deep densely-connected network on top

    if size == "small":
        x = Dense(32, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(12, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(12, activation="relu")(x)

    if size == "medium":
        x = Dense(64, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(12, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(12, activation="relu")(x)

    if size == "large":
        x = Dense(128, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(12, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(12, activation="relu")(x)

    # x = Dropout(rate=dropout)(x)

    # And finally we add the main output layer
    main_output = Dense(1, activation="tanh", name="main_output")(x)

    model = Model(
        inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output]
    )

    model.summary()

    return model


def create_combined_model_conv(
    road_input_shape, aux_input_shape, dropout, recurant_dropout, size
):

    main_input = Input(
        shape=(road_input_shape[0], road_input_shape[1], 1, 1),
        dtype="float32",
        name="road_time_input",
    )

    x = BatchNormalization()(main_input)
    x = ConvLSTM2D(
        filters=32, kernel_size=(5, 1), padding="same", return_sequences=True
    )(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=32, kernel_size=(5, 1), padding="same", return_sequences=False
    )(x)
    x = Dropout(rate=0.1)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    auxiliary_output = Dense(1, activation="tanh", name="aux_output")(x)

    auxiliary_input = Input(shape=(aux_input_shape,), name="aux_input")
    x = keras.layers.concatenate([x, auxiliary_input])

    # We stack a deep densely-connected network on top

    if size == "small":
        x = Dense(32, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(12, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(12, activation="relu")(x)

    if size == "medium":
        x = Dense(64, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(12, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(12, activation="relu")(x)

    if size == "large":
        x = Dense(128, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(12, activation="relu")(x)
        x = Dropout(rate=dropout)(x)
        x = Dense(12, activation="relu")(x)

    # x = Dropout(rate=dropout)(x)

    # And finally we add the main output layer
    main_output = Dense(1, activation="tanh", name="main_output")(x)

    model = Model(
        inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output]
    )

    model.summary()

    return model


def full_connected(
    train_matrix_scaled,
    train_target_scaled,
    test_matrix_scaled,
    days,
    scaler_target,
    dropout,
    size,
    loss,
):
    if size == "small":
        model = create_fully_connected_small(train_matrix.shape[1], dropout)

    if size == "medium":
        model = create_fully_connected_medium(train_matrix.shape[1], dropout)

    if size == "large":
        model = create_fully_connected_large(train_matrix.shape[1], dropout)

    Path(f"Data Exploration/models {days}days").mkdir(parents=True, exist_ok=True)

    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
        keras.callbacks.ModelCheckpoint(
            filepath=f"Data Exploration/models {days}days/full_conn_model_rdrop_int.h5",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.compile(optimizer="rmsprop", loss=loss)
    model.fit(
        train_matrix_scaled,
        train_target_scaled,
        epochs=100,
        callbacks=callbacks_list,
        batch_size=BATCH_SIZE,
        validation_data=(test_matrix_scaled, test_target_scaled),
    )

    test_y_scaled = model.predict(test_matrix_scaled)

    return scaler_target.inverse_transform(test_y_scaled).squeeze()


def lstm(
    train_matrix_st,
    train_target_scaled,
    test_matrix_st,
    days,
    scaler_target,
    dropout,
    recurant_dropout,
    size,
    loss,
):

    if USE_CONV:
        model = create_LSTM_conv(
            (train_matrix_st.shape[1], train_matrix_st.shape[2]),
            dropout,
            recurant_dropout,
        )
    else:

        if size == "small":
            model = create_LSTM_small(
                (train_matrix_st.shape[1], train_matrix_st.shape[2]),
                dropout,
                recurant_dropout,
            )

        if size == "medium":
            model = create_LSTM_medium(
                (train_matrix_st.shape[1], train_matrix_st.shape[2]),
                dropout,
                recurant_dropout,
            )

        if size == "large":
            model = create_LSTM_large(
                (train_matrix_st.shape[1], train_matrix_st.shape[2]),
                dropout,
                recurant_dropout,
            )

    Path(f"Data Exploration/models {days}days").mkdir(parents=True, exist_ok=True)

    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
        keras.callbacks.ModelCheckpoint(
            filepath=f"Data Exploration/models {days}days/lstm_model_rdrop_int.h5",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    if USE_CONV:
        train_matrix_st = train_matrix_st[:, :, :, np.newaxis, np.newaxis]
        test_matrix_st = test_matrix_st[:, :, :, np.newaxis, np.newaxis]

    model.compile(optimizer="rmsprop", loss=loss)
    model.fit(
        train_matrix_st,
        train_target_scaled,
        epochs=100,
        callbacks=callbacks_list,
        batch_size=BATCH_SIZE,
        validation_data=(test_matrix_st, test_target_scaled),
    )

    test_y_scaled = model.predict(test_matrix_st)

    return scaler_target.inverse_transform(test_y_scaled).squeeze()


def combined(
    train_matrix_scaled,
    test_matrix_scaled,
    train_matrix_st,
    train_target_scaled,
    test_matrix_st,
    days,
    scaler_target,
    dropout,
    recurant_dropout,
    intermediate_weight,
    size,
    loss,
):

    if USE_CONV:
        model = create_combined_model_conv(
            (train_matrix_st.shape[1], train_matrix_st.shape[2]),
            train_matrix_scaled.shape[1],
            dropout,
            recurant_dropout,
            size,
        )
    else:
        model = create_combined_model(
            (train_matrix_st.shape[1], train_matrix_st.shape[2]),
            train_matrix_scaled.shape[1],
            dropout,
            recurant_dropout,
            size,
        )

    Path(f"Data Exploration/models {days}days").mkdir(parents=True, exist_ok=True)

    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
        keras.callbacks.ModelCheckpoint(
            filepath=f"Data Exploration/models {days}days/combined_model_rdrop_int.h5",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.compile(
        optimizer="rmsprop",
        loss=loss,
        # metrics=["MAE"],
        loss_weights=[1, intermediate_weight],
    )

    if USE_CONV:
        train_matrix_st = train_matrix_st[:, :, :, np.newaxis, np.newaxis]
        test_matrix_st = test_matrix_st[:, :, :, np.newaxis, np.newaxis]

    model.fit(
        [train_matrix_st, train_matrix_scaled],
        [train_target_scaled, train_target_scaled],
        epochs=100,
        callbacks=callbacks_list,
        batch_size=BATCH_SIZE,
        validation_data=(
            [test_matrix_st, test_matrix_scaled],
            [test_target_scaled, test_target_scaled],
        ),
    )

    test_y_scaled, _ = model.predict([test_matrix_st, test_matrix_scaled])

    return scaler_target.inverse_transform(test_y_scaled).squeeze()


def MAPE(forecast, actual):

    if len(forecast) != len(actual):
        raise ValueError(
            "Could not calculate MAPE, forecast and actual arrays are different length"
        )

    forecast = np.asarray(forecast)
    actual = np.asarray(actual)

    with np.errstate(divide="ignore", invalid="ignore"):

        division = (actual - forecast) / actual

        division[actual == 0] = 0

        # Instead of dividing by n we count by the number of non-zero values.
        # Essentially ignoring all cases where the actual value is zero.
        mape = 100 / np.count_nonzero(actual) * np.sum(np.abs(division))

    return mape


def make_prediction(test, results):
    return test * (1 + (results / 100))


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg  # return a filename


if __name__ == "__main__":

    parser = ArgumentParser(description="Do kitchen sink predictions")
    parser.add_argument(
        "-i",
        dest="input_filename",
        required=True,
        help="input feather file from a previous step",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
    )

    args = parser.parse_args()

    from_path = Path(args.input_filename)

    cols = [
        "index",
        "segment_duration",
        "mean_durations_by_segment_code",
        "mean_durations_by_segment_code_and_hour_and_day",
        "diff_segment_and_mean_by_segment_code",
        "diff_segment_and_mean_by_segment_code_and_hour_and_day",
        "line_distance",
        "to_centre_dist",
        "direction_degrees",
        "rain",
        "median_durations_by_segment_code_and_hour_and_day",
        "arrival_hour",
        "arrival_day",
        "diff_percent_segment_and_mean_by_segment_code_and_hour_and_day",
        "diff_percent_segment_and_median_by_segment_code_and_hour_and_day",
        "date",
        "workid",
        "actualArrival",
        "publicName",
        "test",
        "train",
        "segment_code",
        "clock_direction_degrees",
        "timetable_segment_duration",
        "dry",
        "weekend",
    ]

    if USE_5:

        for i in range(1, 5):
            for j in range(4, 8):
                if i == 1:
                    cols.append(f"self_offset_5_{j}")
                cols.append(f"prev_stop_{i}_offset_5_{j}")
                cols.append(f"next_stop_{i}_offset_5_{j}")

    if USE_10:

        for i in range(1, 8):
            for j in range(3, 15):
                if i == 1:
                    cols.append(f"self_offset_10_{j}")
                cols.append(f"prev_stop_{i}_offset_10_{j}")
                cols.append(f"next_stop_{i}_offset_10_{j}")
        # cols.extend(["prev_stop_4_offset_5_4",
        #     "prev_stop_3_offset_5_4",
        #     "prev_stop_2_offset_5_4",
        #     "prev_stop_1_offset_5_4",
        #     "self_offset_5_4",
        #     "next_stop_1_offset_5_4",
        #     "next_stop_2_offset_5_4",
        #     "next_stop_3_offset_5_4",
        #     "next_stop_4_offset_5_4",
        #     "prev_stop_4_offset_5_5",
        #     "prev_stop_4_offset_5_6",
        #     "prev_stop_4_offset_5_7",
        #     "prev_stop_3_offset_5_5",
        #     "prev_stop_3_offset_5_6",
        #     "prev_stop_3_offset_5_7",
        #     "prev_stop_2_offset_5_5",
        #     "prev_stop_2_offset_5_6",
        #     "prev_stop_2_offset_5_7",
        #     "prev_stop_1_offset_5_5",
        #     "prev_stop_1_offset_5_6",
        #     "prev_stop_1_offset_5_7",
        #     "self_offset_5_5",
        #     "self_offset_5_6",
        #     "self_offset_5_7",
        #     "next_stop_1_offset_5_5",
        #     "next_stop_1_offset_5_6",
        #     "next_stop_1_offset_5_7",
        #     "next_stop_2_offset_5_5",
        #     "next_stop_2_offset_5_6",
        #     "next_stop_2_offset_5_7",
        #     "next_stop_3_offset_5_5",
        #     "next_stop_3_offset_5_6",
        #     "next_stop_3_offset_5_7",
        #     "next_stop_4_offset_5_5",
        #     "next_stop_4_offset_5_6",
        #     "next_stop_4_offset_5_7",])

    print("Loading data...")
    # Load in the stop_events from the previous stage in the pipeline
    stop_events = feather.read_dataframe(args.input_filename, columns=cols)
    stop_events = stop_events.set_index("index")

    se = stop_events

    se["diff_segment_and_mean_by_segment_code"] = (
        se["segment_duration"] - se["mean_durations_by_segment_code"]
    )
    se["diff_segment_and_mean_by_segment_code_and_hour_and_day"] = (
        se["segment_duration"] - se["mean_durations_by_segment_code_and_hour_and_day"]
    )

    se["diff_percent_segment_and_mean_by_segment_code"] = (
        se["diff_segment_and_mean_by_segment_code"]
        * 100
        / se["mean_durations_by_segment_code"]
    )

    se["diff_percent_segment_and_mean_by_segment_code_and_hour_and_day"] = (
        se["diff_segment_and_mean_by_segment_code_and_hour_and_day"]
        * 100
        / se["mean_durations_by_segment_code_and_hour_and_day"]
    )

    segment_code_and_hour_and_day_groups = se[se["train"]].groupby(
        ["segment_code", "arrival_hour", "arrival_day"]
    )

    std_diff_percent_segment_mean_by_segment_code_and_hour_and_day = (
        segment_code_and_hour_and_day_groups[
            "diff_percent_segment_and_mean_by_segment_code_and_hour_and_day"
        ]
        .std()
        .rename("std_diff_percent_segment_mean_by_segment_code_and_hour_and_day")
    )
    se = se.merge(
        std_diff_percent_segment_mean_by_segment_code_and_hour_and_day.to_frame(),
        "left",
        left_on=["segment_code", "arrival_hour", "arrival_day"],
        right_index=True,
    )

    stop_events = se

    if "test" in se.columns:
        print("test in se after loading data")

    # Ensure that the segment code is using the previous
    # timing point not the current one as we use  the previous
    # dwell time.
    # stop_events["segment_code"] = (
    #     stop_events.prev_stopCode
    #     + "_"
    #     + stop_events.stopCode
    #     + "_"
    #     + stop_events.prev_timingPoint.str[0]
    # )

    print("\tLoaded")

    print("Sorting Data...")

    for days in [70, 50, 30, 10, 5, 1]:

        (
            train_matrix,
            train_target,
            test_matrix,
            test_target,
            train_matrix_scaled,
            train_target_scaled,
            test_matrix_scaled,
            test_target_scaled,
            train_matrix_st,
            test_matrix_st,
            scaler_target,
            test_means,
            test_durations,
            test,
        ) = sort_data(stop_events, days)

        print("\tSorted")

        # predict_reg = make_prediction(
        #     test_means, linear_reg(train_matrix, train_target, test_matrix)
        # )

        # predict_xg = make_prediction(
        #     test_means, XGBoost_reg(train_matrix, train_target, test_matrix, days)
        # )

        # predict_rf = make_prediction(
        #     test_means, RF_reg(train_matrix, train_target, test_matrix, days)
        # )

        stats = Stats(test)

        with open(f"size-results-{days}_10_20mins_final_huge_2nd.txt", "w") as file:

            file.write(stats.full_stats(tests=[], names=[], data_type="duration"))
            file.flush()

            dropout = 0
            recurant_dropout = 0.125
            for r_drop in [0]:
                for inter_weight in [0.4]:

                    for loss in ["logcosh", "mean_squared_error"]:
                        # "mean_absolute_error"]:

                        for size in ["small"]:  # "medium", "large"]:

                            predict_NN = make_prediction(
                                test_means,
                                full_connected(
                                    train_matrix_scaled,
                                    train_target_scaled,
                                    test_matrix_scaled,
                                    days,
                                    scaler_target,
                                    dropout,
                                    size,
                                    loss,
                                ),
                            )

                            file.write(
                                str(
                                    stats.single_row(
                                        predict_NN,
                                        f"NN_{size}_{loss}_{str(days)}",
                                        data_type="duration",
                                    )
                                )
                            )
                            file.flush()

                            predict_NN_lstm = make_prediction(
                                test_means,
                                lstm(
                                    train_matrix_st,
                                    train_target_scaled,
                                    test_matrix_st,
                                    days,
                                    scaler_target,
                                    dropout,
                                    r_drop,
                                    size,
                                    loss,
                                ),
                            )

                            file.write(
                                stats.single_row(
                                    predict_NN_lstm,
                                    f"LSTM_{size}_{loss}_{str(days)}_rdrop_{r_drop}",
                                    data_type="duration",
                                )
                            )
                            file.flush()

                            predict_NN_combined = make_prediction(
                                test_means,
                                combined(
                                    train_matrix_scaled,
                                    test_matrix_scaled,
                                    train_matrix_st,
                                    train_target_scaled,
                                    test_matrix_st,
                                    days,
                                    scaler_target,
                                    dropout,
                                    r_drop,
                                    inter_weight,
                                    size,
                                    loss,
                                ),
                            )

                            file.write(
                                stats.single_row(
                                    predict_NN_combined,
                                    f"Combined_{size}_{loss}_{str(days)}_rdrop_{r_drop}_int_{inter_weight}",
                                    data_type="duration",
                                )
                            )
                            file.flush()
