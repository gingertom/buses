import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt


class Stats:
    def __init__(self, data):

        baseline_array_cum, actual_array_cum, baseline_median_array_cum = self._calc_baseline_and_actual(
            data
        )

        self.baseline_array_cum = baseline_array_cum
        self.actual_array_cum = actual_array_cum
        self.baseline_median_array_cum = baseline_median_array_cum
        self.first_20mins_mask = actual_array_cum < 20 * 60
        self.data = data

    def full_stats(self, tests, names, data_type="diff"):

        if data_type not in ["diff", "speed", "duration"]:
            raise ValueError("data_type must be: diff, speed or duration")

        print(" & MAPE & RMSE & MAE & PW10 & MAPE & RMSE & MAE & PW10 \\\\")

        self._stats(
            self.actual_array_cum,
            self.baseline_array_cum,
            "Mean (chd)",
            self.first_20mins_mask,
        )
        self._stats(
            self.actual_array_cum,
            self.baseline_median_array_cum,
            "Median (chd)",
            self.first_20mins_mask,
        )

        for i in range(len(tests)):
            test_array_cum = self._calc_prediction_cum_journeys(
                self.data, test=tests[i], data_type=data_type
            )

            self._stats(
                self.actual_array_cum, test_array_cum, names[i], self.first_20mins_mask
            )

    def draw_time(self, data, names, filename=None, data_type="duration"):

        if data_type not in ["diff", "speed", "duration"]:
            raise ValueError("data_type must be: diff, speed or duration")

        for i in range(len(data)):

            plt.plot(
                self._make_accuracy_matrix_minutes(
                    self._calc_prediction_cum_journeys(
                        self.data, test=data[i], data_type=data_type
                    ),
                    self.actual_array_cum,
                )[0][0, :],
                label=names[i],
            )

        plt.xlim(0, 20)
        plt.title("PW10 Scores By How Far In Advance A Prediction Is")
        plt.xlabel("Time (minutes)")
        plt.ylabel("PW10 Score (%)")
        plt.legend()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
        plt.show()

    def _make_accuracy_matrix_minutes(self, predict, max_threshold=10):

        actual_ints = np.array(self.actual_array_cum / 60).astype(int)

        rows = 1  # int(max_threshold / 10)

        max_a = np.nanmax(self.actual_array_cum) / 60

        accuracies_table = np.empty((rows, int(max_a)))
        drift_table = np.empty((rows, int(max_a)))
        frequency = np.empty(int(max_a))

        print("")

        for i in range(int(max_a)):
            print(".", end="", flush=True)
            mask = actual_ints == i

            frequency[i] = np.count_nonzero(mask)

            for j in range(1, rows + 1):
                accuracy, drift = self._percent_in_x_percent(
                    predict[mask], self.actual_array_cum[mask], j * 10
                )
                accuracies_table[j - 1, i] = accuracy
                drift_table[j - 1, i] = drift

        return accuracies_table, frequency, drift_table

    def _percent_in_x_percent(self, predict, actual, threshold):

        if np.count_nonzero(~np.isnan(actual)) == 0:
            return 0, 0

        threshold = threshold / 100

        mask = ~np.isnan(predict) & ~np.isnan(actual)

        pass_count = np.count_nonzero(
            (predict[mask] < actual[mask] * (1 + threshold))
            & (predict[mask] > actual[mask] * (1 - threshold))
        )

        over_count = np.count_nonzero(predict[mask] > actual[mask] * (1 + threshold))

        under_count = np.count_nonzero(predict[mask] < actual[mask] * (1 - threshold))

        pass_percent = pass_count / np.count_nonzero(mask) * 100

        if over_count + under_count == 0:
            drift = 0.5
        else:
            drift = over_count / (over_count + under_count)

        return pass_percent, drift

    def _stats(self, actual_array_cum, test_array_cum, name, first_20mins_mask):

        mape_short = self._MAPE(test_array_cum[:, 0], actual_array_cum[:, 0])
        mape_long = self._MAPE(test_array_cum, actual_array_cum)
        # rmse_short = np.sqrt(
        #     mean_squared_error(actual_array_cum[:, 0], test_array_cum[:, 0])
        # )
        rmse_short = np.sqrt(
            np.nanmean(np.square(actual_array_cum[:, 0] - test_array_cum[:, 0]))
        )
        rmse_long = np.sqrt(np.nanmean(np.square(actual_array_cum - test_array_cum)))
        # mae_short = mean_absolute_error(actual_array_cum[:, 0], test_array_cum[:, 0])
        mae_short = np.nanmean(np.abs(actual_array_cum[:, 0] - test_array_cum[:, 0]))
        mae_long = np.nanmean(np.abs(actual_array_cum - test_array_cum))

        pass_count_short = np.count_nonzero(
            (test_array_cum[:, 0] < actual_array_cum[:, 0] * 1.1)
            & (test_array_cum[:, 0] > actual_array_cum[:, 0] * 0.9)
        )

        pass_fraction_short = pass_count_short / actual_array_cum.shape[0]

        pass_count_long = np.count_nonzero(
            (
                test_array_cum[first_20mins_mask]
                < actual_array_cum[first_20mins_mask] * 1.1
            )
            & (
                test_array_cum[first_20mins_mask]
                > actual_array_cum[first_20mins_mask] * 0.9
            )
        )
        pass_fraction_long = pass_count_long / np.count_nonzero(first_20mins_mask)

        print(
            f"{name} & {mape_short:0.3f} & {rmse_short:0.3f} & {mae_short:0.3f} & {pass_fraction_short*100:0.3f} & {mape_long:0.3f} & {rmse_long:0.3f} & {mae_long:0.3f} & {pass_fraction_long*100:0.3f} \\\\"
        )

    def _MAPE(self, forecast, actual):

        if len(forecast) != len(actual):
            raise ValueError(
                "Could not calculate MAPE, forecast and actual arrays are different length"
            )

        forecast = np.asarray(forecast)
        actual = np.asarray(actual)

        with np.errstate(divide="ignore", invalid="ignore"):

            division = (actual - forecast) / actual

            division[actual == 0] = 0

            valid_count = actual.size - np.count_nonzero(
                (actual == 0) | (np.isnan(actual))
            )

            # Instead of dividing by n we count by the number of non-zero and non-nan values.
            # Essentially ignoring all cases where the actual value is zero.
            mape = 100 / valid_count * np.nansum(np.abs(division))

        return mape

    def _create_padded_array(self, a, row_start, n_rows):
        # From: https://stackoverflow.com/questions/51597849/padding-a-numpy-array-with-offsets-for-each-data-column

        r = np.arange(n_rows)[:, None]
        row_start = np.asarray(row_start)
        mask = (r >= row_start) & (r < row_start + a.shape[0])

        out = np.zeros(mask.shape, dtype=a.dtype)
        out[:] = np.nan
        out.T[mask.T] = a.ravel("F")
        return out

    def _create_triangle(self, input_array, max_width=70):

        filled_values = np.empty((input_array.shape[0], 70)).astype(float)

        filled_values[:] = input_array[:, None]

        return self._create_padded_array(
            filled_values, list(range(70)), input_array.shape[0] + 71
        )[: input_array.shape[0], :70]

    def _calc_baseline_and_actual(self, data):

        se_min = data.copy()

        baseline_array = np.empty((se_min.shape[0], 70)).astype(float)
        baseline_array[:] = np.nan

        actual_array = np.empty((se_min.shape[0], 70)).astype(float)
        actual_array[:] = np.nan

        baseline_median_array = np.empty((se_min.shape[0], 70)).astype(float)
        baseline_median_array[:] = np.nan

        se_min = se_min.reset_index(drop=True)

        runs = se_min.groupby(["date", "workid"])

        actual_index = se_min.columns.get_loc("segment_duration")
        baseline_index = se_min.columns.get_loc(
            "mean_durations_by_segment_code_and_hour_and_day"
        )
        baseline_median_index = se_min.columns.get_loc(
            "median_durations_by_segment_code_and_hour_and_day"
        )

        for _, run in runs:
            run = run.sort_values("actualArrival")

            baseline_array[run.index, :] = self._create_triangle(
                run.iloc[:, baseline_index]
            )
            actual_array[run.index, :] = self._create_triangle(
                run.iloc[:, actual_index]
            )
            baseline_median_array[run.index, :] = self._create_triangle(
                run.iloc[:, baseline_median_index]
            )

        baseline_array_cum = np.cumsum(baseline_array, axis=1)
        actual_array_cum = np.cumsum(actual_array, axis=1)
        baseline_median_array_cum = np.cumsum(baseline_median_array, axis=1)

        actual_array_cum = np.clip(actual_array_cum, 0, 2 * 60 * 60)

        return baseline_array_cum, actual_array_cum, baseline_median_array_cum

    def _calc_prediction_cum_journeys(self, data, test, data_type="diff"):

        se_min = data[
            [
                "date",
                "workid",
                "actualArrival",
                "mean_durations_by_segment_code_and_hour_and_day",
            ]
        ].copy()

        if data_type == "diff":
            se_min["prediction"] = se_min[
                "mean_durations_by_segment_code_and_hour_and_day"
            ] * (1 + (test / 100))

        if data_type == "speed":
            se_min["prediction"] = se_min["real_length"] / test * 2.237

        if data_type == "duration":
            se_min["prediction"] = test

        predict_array = np.empty((se_min.shape[0], 70)).astype(float)
        predict_array[:] = np.nan

        se_min = se_min.reset_index(drop=True)

        runs = se_min.groupby(["date", "workid"])

        prediction_index = se_min.columns.get_loc("prediction")

        for _, run in runs:
            run = run.sort_values("actualArrival")

            predict_array[run.index, :] = self._create_triangle(
                run.iloc[:, prediction_index]
            )

        predict_array_cum = np.cumsum(predict_array, axis=1)

        return predict_array_cum
