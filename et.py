import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from kneed import KneeLocator
import random

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
            plt.xlim(-10, 20)
            plt.ylim(-0.1, 1.1)
            plt.title("Threshold: " + str(round(self.threshold, 2)) + " Lines: " + str(len(self.dx)))
            
        plt.show()

    def set_threshold(self, threshold):
        self.threshold = threshold