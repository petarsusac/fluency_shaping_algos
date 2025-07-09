from plot import PlotData
import numpy as np

def speech_rate_warning(features: PlotData, threshold_rate=150, threshold_frames=3):
    rate_list = np.array(features.rate_list)
    if len(rate_list) > 0:
        if np.sum(rate_list > threshold_rate) >= threshold_frames:
            return True
    
    return False

def phonation_warning(features: PlotData, threshold_duration=20, threshold_num=1):
    short_intervals = 0
    if features.phonation_intervals:
        for interval in features.phonation_intervals:
            if (interval[1] - interval[0]) < threshold_duration and features.speech_activity[interval[0]:interval[1]].any() \
                and interval[0] != 0 and interval[1] != len(features.zcr):
                short_intervals += 1
    return short_intervals >= threshold_num