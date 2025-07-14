import numpy as np
import pyaudio
import librosa
import scipy.signal
import threading
from vad import VADPowerThreshold
from imu_respiration import IMURespiration, IMUNotFoundError
from onset import get_hard_onsets
from speech_rate import signal_power_db, speech_rate_estimate_power
from plot import Plot, PlotData
from queue import Queue

PROC_FRAME_LEN=2048
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
        respiration_sample = imu.get_sample()
        if respiration_sample is not None:
            respiration_filtered.pop(0)
            respiration_filtered.append(respiration_sample)


def init_globals():
    global plotting_queue
    plotting_queue = Queue()

    global continue_recording, chunk_idx, chunks, noise_power, rate_list, smoothed_rate_list, zcr_b, zcr_a, audio_hp_b, audio_hp_a, speech_rate_estimate, speech_timestamps, speech_activity, power, zcr, hard_onsets, phonation_intervals
    continue_recording = True
    chunk_idx = 0
    noise_power = 100
    chunks = [np.zeros(CHUNK) for _ in range((FRAME_LEN_SEC * RATE) // CHUNK)]
    power = signal_power_db(np.concatenate(chunks), frame_length=PROC_FRAME_LEN, hop=PROC_HOP_LEN)
    zcr = librosa.feature.zero_crossing_rate(y=np.concatenate(chunks), frame_length=PROC_FRAME_LEN, hop_length=PROC_HOP_LEN)[0]
    zcr_b, zcr_a = scipy.signal.butter(4, 0.5, 'low')
    audio_hp_b, audio_hp_a = scipy.signal.butter(4, Wn=100, fs=16000, btype='high')
    rate_list = [0] * 10
    smoothed_rate_list = [0] * 10
    speech_rate_estimate = speech_rate_estimate_power(power)
    speech_timestamps = []
    speech_activity = np.zeros(len(power))
    hard_onsets = []
    phonation_intervals = []

    global vad
    vad = VADPowerThreshold(threshold=100)

    global respiration_filtered, imu_present, imu
    respiration_filtered = [0.0] * FRAME_LEN_SEC * 10
    imu_present = True
    try:
        imu = IMURespiration()
    except IMUNotFoundError:
        print("IMU not found. Respiration acquisition will be skipped.")
        imu_present = False

    global stop_thread, serial_thread
    stop_thread = threading.Thread(target=stop)
    stop_thread.start()

    if imu_present:
        serial_thread = threading.Thread(target=acq_respiration)
        serial_thread.start()

    plot_init_data = PlotData(
        rate_list=smoothed_rate_list,
        power=power,
        speech_activity=speech_activity,
        speech_rate_estimate=speech_rate_estimate,
        zcr=zcr,
        zcr_threshold=ZCR_THRESHOLD,
        hard_onsets=hard_onsets,
        phonation_intervals=phonation_intervals,
        respiration_filtered=respiration_filtered
    )

    global plot
    plot = Plot(plot_init_data)

def audio_process(in_data, frame_count, time_info, status):
    global chunk_idx, noise_power
    chunk_idx += 1
    chunk = np.frombuffer(in_data, dtype=np.float32)
    chunks.pop(0)
    chunks.append(chunk)

    audio = np.concatenate(chunks)
    audio = scipy.signal.filtfilt(audio_hp_b, audio_hp_a, audio)
    power = signal_power_db(audio, frame_length=PROC_FRAME_LEN, hop=PROC_HOP_LEN)
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=PROC_FRAME_LEN, hop_length=PROC_HOP_LEN)[0]
    if chunk_idx == 10:
        print(audio.shape, power.shape, zcr.shape)
        noise_power = np.mean(power)
        vad.set_threshold(noise_power + 5)
        print(f"Noise power estimated: {noise_power:.2f} dB")

    speech_rate_estimate = speech_rate_estimate_power(power, peak_th=noise_power+5, peak_prominence=2)
    zcr = scipy.signal.filtfilt(zcr_b, zcr_a, zcr)

    speech_timestamps, speech_activity = vad.process(power)
    num_syllables = np.sum(speech_activity[speech_rate_estimate["peaks"]])
    if (np.sum(speech_activity) > 0):
        speech_rate = num_syllables * (60 / FRAME_LEN_SEC)
    else:
        speech_rate = 0

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
    # window_size = 3
    # smoothed_speech_rate = np.convolve(rate_list[-window_size:], np.ones(window_size)/window_size, mode='valid')[0]
    smoothed_rate_list.pop(0)
    smoothed_rate_list.append(speech_rate)

    plotting_data = PlotData(
        rate_list=smoothed_rate_list,
        power=power,
        speech_activity=speech_activity,
        speech_rate_estimate=speech_rate_estimate,
        zcr=zcr,
        hard_onsets=hard_onsets,
        phonation_intervals=phonation_intervals,
        respiration_filtered=respiration_filtered
    )

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
        imu.close()

if __name__ == "__main__":
    main()