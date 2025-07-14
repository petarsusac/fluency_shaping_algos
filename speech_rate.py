import numpy as np
import scipy.signal

def signal_power(signal, frame_length, hop):
    power = np.zeros((len(signal) - frame_length) // hop + 1)
    for i in range(0, len(signal), hop):
        frame = signal[i:i + frame_length]
        frame_idx = i // hop
        if len(frame) < frame_length:
            break
        power[frame_idx] = np.sum((frame / 2e-5) ** 2) / frame_length
    return power

def signal_power_db(signal, frame_length, hop):
    return 10 * np.log10(signal_power(signal, frame_length, hop))

def signal_amplitude(signal, frame_length, hop):
    amplitude = np.zeros((len(signal) - frame_length) // hop + 1)
    for i in range(0, len(signal), hop):
        frame = signal[i:i + frame_length]
        frame_idx = i // hop
        if len(frame) < frame_length:
            break
        amplitude[frame_idx] = np.sum(np.abs(frame))
    return amplitude

def speech_rate_estimate_power(power, peak_th=-40, peak_prominence=2):
    peaks_power = scipy.signal.find_peaks(power, height=peak_th, prominence=peak_prominence)[0]
    return {
        "num_syllables": peaks_power.shape[0],
        "peaks": peaks_power,
    }