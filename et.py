from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from kneed import KneeLocator
from utils import Anomaly
from datetime import timedelta

def linear_interpolation(x, x0, y0, x1, y1):
    slope = (y1 - y0) / (x1 - x0)
    return y0 + (x - x0) * slope

class ET:
    def __init__(self, dx, dy):
        if len(dx) != len(dy):
            raise Exception("ET curve must have same number of points in dx and dy")
        
        if len(dx) < 2:
            raise Exception("At least two points is required to create ET curve")
        
        self.dx = dx
        self.dy = dy
        self.threshold = None

    def expected(self, temperature):
        if temperature < self.dx[0]:
            return linear_interpolation(temperature, self.dx[0], self.dy[0], self.dx[1], self.dy[1])
        if temperature > self.dx[-1]:
            return self.dy[-1]

        for i in range(len(self.dx) - 1):
            if self.dx[i] <= temperature and temperature <= self.dx[i+1]:
                return linear_interpolation(temperature, self.dx[i], self.dy[i], self.dx[i+1], self.dy[i+1])
    
    def get_expected_series(self, temperature_series):
        return temperature_series.apply(self.expected)
    
    def get_proportial_series(self, energy_series, temperature_series):
        expected_series = self.get_expected_series(temperature_series)
        return energy_series / expected_series

    def get_anomolies_series(self, energy_series, temperature_series):
        if self.threshold is None:
            raise Exception("Threshold not set. Use set_threshold()")
        expected_series = self.get_expected_series(temperature_series)
        mask = (energy_series - expected_series).abs() >= self.threshold
        return energy_series[mask]

    def get_anomolies(self, energy_series, temperature_series, building_name=''):
        anomalies_series = self.get_anomolies_series(energy_series, temperature_series)

        anomalies = []

        for anomaly_start in anomalies_series.index:
            anomaly_end = anomaly_start + timedelta(days=1)
            anomaly = Anomaly(building_name, anomaly_start, anomaly_end)
            anomalies.append(anomaly)
        
        print(f'anomalies_series len: {len(anomalies_series)}\n anomalies len: {len(anomalies)}')

        return anomalies

    def get_mse(self, energy_series, temperature_series):
        expected_series = self.get_expected_series(temperature_series)
        return ((energy_series - expected_series) ** 2).mean()

    def get_top_diffs(self, energy_series, temperature_series, n):
        expected_series = self.get_expected_series(temperature_series)
        diffs = (energy_series - expected_series).abs()
        # sort diffs in descending order
        sorted_diffs = diffs.sort_values(ascending=False)
        # get the top n values
        top_n = sorted_diffs[:n]
        return top_n

    def plot(self, energy_series, temperature_series):
        plt.plot(self.dx, self.dy, color='red', linewidth=4)

        # if threshold is set, find anomalies and color them in the scatter plot
        plt.scatter(temperature_series, energy_series, color='blue')
        if self.threshold is not None:
            anomalies = self.get_anomolies_series(energy_series, temperature_series)
            plt.scatter(temperature_series[anomalies.index], energy_series[anomalies.index], color='red', s=100)
            # plot one curve over and under the expected curve
            sorted_temperature_series = temperature_series.sort_values()
            plt.plot(sorted_temperature_series, self.get_expected_series(sorted_temperature_series) + self.threshold, color='green', linewidth=3)
            plt.plot(sorted_temperature_series, self.get_expected_series(sorted_temperature_series) - self.threshold, color='green', linewidth=3)
            # plt.xlim(-10, 20)
            # plt.ylim(-0.1, 1.1)
            plt.title(f'Threshold: {self.threshold:.2f}  Lines: {len(self.dx)}')

        plt.show()

    def set_threshold(self, threshold):
        self.threshold = threshold

class ETT:
    def __init__(self):
        self.ETs: Dict[ET] = {}

        self.week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.lines = range(1,8) # num of lines to test ET on

        self.best_line_amounts = []
        self.best_anomaly_amounts = []
        self.best_mse_amounts = []
        self.best_thresholds = []

    def optimizer(self, energy_consumption_series, temperature_series) -> bool:
        for day_index, day in enumerate(self.week_days):
            weekday_energy = energy_consumption_series.loc[energy_consumption_series.index.weekday == day_index]
            weekday_temp = temperature_series.loc[temperature_series.index.weekday == day_index]

            # find best amount of lines to use this weekday
            lines_mse = []
            for line in self.lines:
                pps = len(weekday_energy) // line + 1
                et = regressor(weekday_energy, weekday_temp, pps)

                if et is None:
                    continue

                lines_mse.append(et.get_mse(weekday_energy, weekday_temp))

            if not lines_mse:
                return None

            # use knee method to find best amount of lines and pps
            not_knee = KneeLocator(self.lines, lines_mse, curve='convex', direction='decreasing')

            best_line_amount = not_knee.knee
            if best_line_amount is None:
                best_line_amount = self.lines[-1]
            best_pps = len(weekday_energy) // best_line_amount + 1

            # recreate ET with best amount of lines
            et = regressor(weekday_energy, weekday_temp, best_pps)
            # get all residuals sorted in descending order
            top_diffs = et.get_top_diffs(weekday_energy, weekday_temp, len(weekday_energy))
            top_diffs.dropna(inplace=True)

            # find the best amount of anomalies to include for the data
            not_knee = KneeLocator(np.arange(1, len(top_diffs) + 1), top_diffs, curve='convex', direction='decreasing')
            best_anomaly_amount = not_knee.knee
            if best_anomaly_amount is None:
                print('did not fint best anomaly amount')
                return False

            # find the corresponding threshold
            best_threshold = top_diffs.values[best_anomaly_amount - 1]
            et.set_threshold(best_threshold)
            self.ETs[day] = et

            # save some stats
            self.best_line_amounts.append(best_line_amount)
            self.best_anomaly_amounts.append(best_anomaly_amount)
            self.best_mse_amounts.append(et.get_mse(weekday_energy, weekday_temp))
            self.best_thresholds.append(best_threshold)
        return True

    def expected(self, temperature, day_of_week):
        return self.ETs[day_of_week].expected(temperature)

    def get_expected_series(self, temperature_series: pd.Series):
        d = temperature_series.index.dayofweek
        temperature_series_split_day = [temperature_series.iloc[d == x] for x in range(7)]

        h = [t.apply(self.expected) for t in temperature_series_split_day]

        expected = pd.concat(h, axis='index')
        expected.sort_index(inplace=True)
        return temperature_series.apply(self.expected)
    
    def get_anomalies_series(self, energy_series, temperature_series):
        e_i = energy_series.index.dayofweek
        t_i = temperature_series.index.dayofweek
        anomalies_series_list = []
        for i in range(7):
            e = energy_series.iloc[e_i == i]
            t = temperature_series.iloc[t_i == i]
            anomalies_series_list.append(self.ETs[self.week_days[i]].get_anomolies_series(e, t))
        anomalies_series = pd.concat(anomalies_series_list)
        anomalies_series.sort_index(inplace=True)
        return anomalies_series


    def get_anomalies(self, energy_series, temperature_series, building_name=''):
        anomalies_series = self.get_anomalies_series(energy_series, temperature_series)

        anomalies = []

        for anomaly_start in anomalies_series.index:
            anomaly_end = anomaly_start + timedelta(days=1)
            anomaly = Anomaly(building_name, anomaly_start, anomaly_end)
            anomalies.append(anomaly)
        
        # print(f'anomalies_series len: {len(anomalies_series)}\n anomalies len: {len(anomalies)}')

        return anomalies
    
    def plot(self, energy_series: pd.Series, temperature_series: pd.Series):
        e_i = energy_series.index.dayofweek
        t_i = temperature_series.index.dayofweek
        for i in range(7):
            print(f'{self.week_days[i]}')
            e = energy_series.iloc[e_i == i]
            t = temperature_series.iloc[t_i == i]
            self.ETs[self.week_days[i]].plot(e, t)

def regressor(energy_consumption_series, temperature_series, points_per_segment):
    # ensure that we match values for energy and temp
    energy_consumption_series.rename('energy', inplace=True)
    temperature_series.rename('temperature', inplace=True)
    df = pd.concat((energy_consumption_series, temperature_series), axis=1, ignore_index=False)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.sort_values(by='temperature', inplace=True)

    number_of_points = df.shape[0]

    X = df['temperature'].values.reshape(number_of_points, 1)
    y = df['energy'].values.reshape(number_of_points, 1)

    dx = []
    dy = []

    for i in range(0, number_of_points, points_per_segment):
        n = min(points_per_segment, number_of_points - i)

        X_seg = X[i:i+n, :]
        y_seg = y[i:i+n, :]

        model = LinearRegression()
        model.fit(X_seg, y_seg)

        if i == 0:
            X_first = X_seg[0].reshape(1,1)
            y_first = model.predict(X_first)
            dx.append(X_first[0,0])
            dy.append(y_first[0,0])

        X_middle = X_seg[n//2].reshape(1,1)
        y_middle = model.predict(X_middle)
        dx.append(X_middle[0,0])
        dy.append(y_middle[0,0])

        if i >= number_of_points - points_per_segment:
            X_last = X_seg[-1].reshape(1,1)
            y_last = model.predict(X_last)
            dx.append(X_last[0,0])
            dy.append(y_last[0,0])

    et = None
    try:
        et = ET(dx, dy)
    except:
        print('failed to create valid et curve')

    return et
