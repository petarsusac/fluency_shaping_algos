import numpy as np
import scipy.signal

def signal_power(signal, frame_length, hop):
    power = np.zeros((len(signal) - frame_length) // hop + 1)
    for i in range(0, len(signal), hop):
        frame = signal[i:i + frame_length]
        frame_idx = i // hop
        if len(frame) < frame_length:
            break
        power[frame_idx] = np.sum(frame ** 2) / frame_length
    return power

def signal_power_db(signal, frame_length, hop):
    return 10 * np.log10(signal_power(signal, frame_length, hop))

def speech_rate_estimate_power(power, zcr):
    peaks_power = scipy.signal.find_peaks(power, height=-60, prominence=2)[0]
    unvoiced_peaks_idxs = []
    for i, peak in enumerate(peaks_power):
        if zcr[peak] > 0.4:
            unvoiced_peaks_idxs.append(i)
    peaks_power = np.delete(peaks_power, unvoiced_peaks_idxs)
    return {
        "num_syllables": peaks_power.shape[0],
        "peaks": peaks_power,
    }