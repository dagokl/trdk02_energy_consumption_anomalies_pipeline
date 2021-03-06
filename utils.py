import numpy as np

class Anomaly:
    def __init__(self, building, start: np.datetime64, end: np.datetime64, intensity: float):
        self.building = building
        self.start: np.datetime64 = start
        self.end: np.datetime64 = end
        self.intensity = intensity
