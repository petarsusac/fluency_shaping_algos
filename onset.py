import numpy as np

def rise_slope(signal):
    # Signal is in dB, so values closer to 0 should be greater
    signal = 100 + signal
    signal = signal - np.min(signal)

    # Find the max
    max_idx = np.argmax(signal)
    i = max_idx

    # Search to the left for the 90% point
    while (signal[i] > 0.9 * signal[max_idx]) and (i > 0):
        i -= 1

    right = i
    
    # Search to the left for the 10% point
    while (signal[i] > 0.1 * signal[max_idx]) and (i > 0):
        i -= 1

    left = i

    if right != left:
        return (signal[right] - signal[left]) / (right - left)
    else:
        return np.inf

def get_hard_onsets(speech_timestamps, signal_power, threshold=5):
    hard_onsets = []
    for timestamp in speech_timestamps:
        if (timestamp["start"] != 1):
            # TODO: choose padding and treshold
            power_start = timestamp["start"] - 6 # cca 100 ms
            power_end = power_start + 30 # cca 500 ms
            if power_end < len(signal_power):
                power_segment = signal_power[power_start:power_end]
                if len(power_segment) > 0:
                    slope = rise_slope(power_segment)
                    if slope > threshold:
                        hard_onsets.append(timestamp["start"])

    return hard_onsets
