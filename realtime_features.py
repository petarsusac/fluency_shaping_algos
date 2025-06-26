import numpy as np
import pyaudio
import librosa
import scipy.signal
import threading
from vad import VADPowerThreshold
from imu_respiration import imu_respiration_init, imu_respiration_cleanup, get_respiration_sample
from onset import get_hard_onsets
from speech_rate import signal_power_db, speech_rate_estimate_power
from plot import Plot
from queue import Queue

PROC_FRAME_LEN=1024
PROC_HOP_LEN=256
FRAME_LEN_SEC=5
ZCR_THRESHOLD = 0.15
RATE = 16000
CHUNK = 8000
FORMAT = pyaudio.paFloat32
CHANNELS = 1

def stop():
    input("Press Enter to stop:")
    global continue_recording
    continue_recording = False

def acq_respiration():
    while continue_recording:
        respiration_sample = get_respiration_sample()
        if respiration_sample is not None:
            respiration_filtered.pop(0)
            respiration_filtered.append(respiration_sample)


def init_globals():
    global plotting_queue
    plotting_queue = Queue()

    global continue_recording, chunk_idx, chunks, chunks_power, noise_power, chunks_zcr, rate_list, smoothed_rate_list, zcr_b, zcr_a, audio_hp_b, audio_hp_a, pow_lp_b, pow_lp_a, speech_rate_estimate, speech_timestamps, speech_activity, power, zcr, hard_onsets, phonation_intervals
    continue_recording = True
    chunk_idx = 0
    noise_power = 0
    chunks = [np.zeros(CHUNK) for _ in range((FRAME_LEN_SEC * RATE) // CHUNK)]
    chunks_power = [signal_power_db(chunk, frame_length=PROC_FRAME_LEN, hop=PROC_HOP_LEN) for chunk in chunks]
    chunks_zcr = [librosa.feature.zero_crossing_rate(y=chunk, frame_length=PROC_FRAME_LEN, hop_length=PROC_HOP_LEN)[0] for chunk in chunks]
    power = np.concatenate(chunks_power)
    zcr = np.concatenate(chunks_zcr)
    zcr_b, zcr_a = scipy.signal.butter(4, 0.5, 'low')
    audio_hp_b, audio_hp_a = scipy.signal.butter(4, Wn=100, fs=16000, btype='high')
    pow_lp_b, pow_lp_a = scipy.signal.butter(4, Wn=0.15, btype='low')
    rate_list = [0] * 20
    smoothed_rate_list = [0] * 20
    speech_rate_estimate = speech_rate_estimate_power(power, zcr)
    speech_timestamps = []
    speech_activity = np.zeros(len(power))
    hard_onsets = []
    phonation_intervals = []

    global vad
    vad = VADPowerThreshold(threshold_db=0, hangover=20, start_idx=25)

    global respiration_filtered, imu_present
    respiration_filtered = [0.0] * FRAME_LEN_SEC * 10
    imu_present = imu_respiration_init()

    global stop_thread, serial_thread
    stop_thread = threading.Thread(target=stop)
    stop_thread.start()

    if imu_present:
        serial_thread = threading.Thread(target=acq_respiration)
        serial_thread.start()

    plot_init_data = {
        "rate_list": rate_list,
        "power": power,
        "speech_activity": speech_activity,
        "speech_rate_estimate": speech_rate_estimate,
        "zcr": zcr,
        "respiration_filtered": respiration_filtered,
        "zcr_threshold": ZCR_THRESHOLD
    }

    global plot
    plot = Plot(plot_init_data)

def audio_process(in_data, frame_count, time_info, status):
    global chunk_idx, noise_power
    chunk_idx += 1
    chunk = np.frombuffer(in_data, dtype=np.float32)
    chunks.pop(0)
    chunks.append(chunk)
    chunk = scipy.signal.lfilter(audio_hp_b, audio_hp_a, chunk)
    audio_frame = np.concatenate(chunks)
    
    chunks_power.pop(0)
    chunks_power.append(signal_power_db(chunk, frame_length=PROC_FRAME_LEN, hop=PROC_HOP_LEN))
    chunks_zcr.pop(0)
    chunks_zcr.append(librosa.feature.zero_crossing_rate(y=chunk, frame_length=PROC_FRAME_LEN, hop_length=PROC_HOP_LEN)[0])

    power = np.concatenate(chunks_power)
    power = scipy.signal.filtfilt(pow_lp_b, pow_lp_a, power)
    zcr = np.concatenate(chunks_zcr)
    if chunk_idx == 4:
        # Estimate noise power from the first 4 chunks (2 seconds)
        noise_power = np.mean(np.concatenate(chunks_power[-4:]))
        vad.set_threshold(noise_power + 10)

    speech_rate_estimate = speech_rate_estimate_power(power, zcr, peak_th=noise_power+10)
    zcr = scipy.signal.filtfilt(zcr_b, zcr_a, zcr)

    speech_rate = speech_rate_estimate["num_syllables"] * (60 // FRAME_LEN_SEC)
    speech_timestamps, speech_activity = vad.process(power)

    # Onset
    hard_onsets = get_hard_onsets(speech_timestamps, power, threshold=4)

    # Phonation intervals
    phonation_intervals=[]
    i = 0
    while i < len(zcr):
        j = i + 1
        while j < len(zcr) and zcr[j] < ZCR_THRESHOLD:
            j += 1
        if j - i > 5:
            phonation_intervals.append((i, j))
        i = j
        
    rate_list.pop(0)
    rate_list.append(speech_rate)

    # Moving average filter
    window_size = 3
    smoothed_speech_rate = np.convolve(rate_list[-window_size:], np.ones(window_size)/window_size, mode='valid')[0]
    smoothed_rate_list.pop(0)
    smoothed_rate_list.append(smoothed_speech_rate)

    plotting_data = {
        "rate_list": smoothed_rate_list,
        "power": power,
        "speech_activity": speech_activity,
        "speech_rate_estimate": speech_rate_estimate,
        "zcr": zcr,
        "respiration_filtered": respiration_filtered,
        "hard_onsets": hard_onsets,
        "phonation_intervals": phonation_intervals
    }

    plotting_queue.put_nowait(plotting_data)

    if not continue_recording:
        return (None, pyaudio.paComplete)
    return (in_data, pyaudio.paContinue)

def main():
    init_globals()

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_process)

    while(continue_recording):
        plot.update(plotting_queue.get())
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    plot.close()
    stop_thread.join()
    if imu_present:
        serial_thread.join()
    imu_respiration_cleanup()

if __name__ == "__main__":
    main()