import silero_vad
import numpy as np

USE_SILERO = False

if USE_SILERO:
    vad = silero_vad.load_silero_vad()

def vad_power_thresholding(signal_power_db, threshold_db=-60, hangover=5):
    speech_timestamps = []
    power_thresholded = np.array(signal_power_db >= threshold_db).astype(int)
    power_thresholded_hangover = np.zeros(len(power_thresholded))
    for i in range(1, len(power_thresholded)):
        if power_thresholded[i - min(i, hangover):i].any():
            power_thresholded_hangover[i] = 1
        else:
            power_thresholded_hangover[i] = 0

    start = -1
    end = -1
    for i in range(0, len(power_thresholded)):
        if (power_thresholded_hangover[i] == 1 and i == 0):
            start = i
        elif i != 0 and (power_thresholded_hangover[i] == 1 and power_thresholded_hangover[i - 1] == 0):
            start = i
        
        if i != 0 and ((power_thresholded_hangover[i] == 0 and power_thresholded_hangover[i - 1] == 1) 
                       or (power_thresholded_hangover[i] == 1 and i == len(power_thresholded_hangover) - 1)):
            end = i
            speech_timestamps.append({"start": start, "end": end})
    return speech_timestamps, power_thresholded_hangover

def vad_silero(audio, signal_power_len, proc_hop_len):
    if not USE_SILERO:
        raise ValueError("Silero VAD is disabled. Set USE_SILERO to True")
    
    speech_timestamps = silero_vad.get_speech_timestamps(audio, vad, min_silence_duration_ms=500, threshold=0.3)
    speech_activity = np.zeros(signal_power_len)
    for timestamp in speech_timestamps:
        timestamp['start'] = timestamp['start'] // proc_hop_len
        timestamp['end'] = timestamp['end'] // proc_hop_len
        speech_activity[timestamp['start']:timestamp['end']] = 1

    return speech_timestamps, speech_activity
