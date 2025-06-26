import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import librosa
import matplotlib
import scipy.signal
import threading
from vad import vad_silero, vad_power_thresholding, USE_SILERO
from imu_respiration import imu_respiration_init, imu_respiration_cleanup, get_respiration_sample
from onset import get_hard_onsets
from speech_rate import signal_power_db, speech_rate_estimate_power

matplotlib.use('TkAgg')

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
    global continue_recording, chunk_idx
    global chunks, chunks_power, noise_power, chunks_zcr, rate_list, smoothed_rate_list, zcr_b, zcr_a, audio_hp_b, audio_hp_a, pow_lp_b, pow_lp_a, speech_rate_estimate, speech_timestamps, speech_activity, power, zcr, hard_onsets, phonation_intervals
    global fig, line1, line2, line3, line2_peaks, line2_vertical_lines, line2_activity, line3_vertical_lines, line4, line4_activity, ax1, ax2, ax3, ax4
    global ser, ahrs, respiration_filtered
    global stop_thread, serial_thread, imu_present

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
    respiration_filtered = [0.0] * FRAME_LEN_SEC * 10
    imu_present = imu_respiration_init()

    stop_thread = threading.Thread(target=stop)
    stop_thread.start()

    if imu_present:
        serial_thread = threading.Thread(target=acq_respiration)
        serial_thread.start()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    line1, = ax1.plot(rate_list, color="C0")
    line2, = ax2.plot(power, color="C0")
    line2_activity, = ax2.step(np.arange(len(speech_activity)), speech_activity, color='k')
    line2_peaks, = ax2.plot(speech_rate_estimate["peaks"], power[speech_rate_estimate["peaks"]], 'ro')
    line2_vertical_lines = []
    line3, = ax3.plot(zcr, color="C0")
    line3_vertical_lines = []
    line4, = ax4.plot(respiration_filtered)
    line4_activity, = ax4.step(np.arange(len(speech_activity)) / (len(speech_activity) / len(respiration_filtered)), speech_activity, color='k')
    ax1.set_ylim(0, 300)
    ax2.set_ylim(-80, 5)
    ax2.set_xlim(0, len(power))
    ax3.set_ylim(0, 0.4)
    ax1.set_title('Speech rate (syllables/min)')
    ax2.set_title('Signal power (dB)')
    ax3.set_title('Zero crossing rate (voicedness)')
    ax3.axhline(y=ZCR_THRESHOLD, color="black", linestyle='--')
    ax3.set_xlabel('Time (s)')
    ax1.set_ylabel('Syllables/min')
    ax2.set_ylabel('Power (dB)')
    ax3.set_ylabel('ZCR')
    ax4.set_title('Respiration')
    ax4.set_ylim(-1, 1)
    ax4.set_xlim(0, len(respiration_filtered))

    plt.show(block=False)

def audio_process(in_data, frame_count, time_info, status):
    global chunks, rate_list, smoothed_rate_list, speech_rate_estimate, speech_timestamps, speech_activity, zcr, hard_onsets, phonation_intervals, power, zcr
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

    speech_rate_estimate = speech_rate_estimate_power(power, zcr, peak_th=noise_power+10)
    zcr = scipy.signal.filtfilt(zcr_b, zcr_a, zcr)
    hard_onsets = []
    phonation_intervals=[]

    if USE_SILERO:
        num_syllables = 0
        speech_duration_s = FRAME_LEN_SEC
        speech_timestamps, speech_activity = vad_silero(audio_frame, len(power), PROC_HOP_LEN)
        for peak in speech_rate_estimate['peaks']:
            for timestamp in speech_timestamps:
                if peak >= timestamp['start'] and peak <= timestamp['end']:
                    num_syllables += 1
        if speech_duration_s > 0:
            speech_rate = num_syllables / speech_duration_s * 60
        else:
            speech_rate = 0
    else:
        speech_rate = speech_rate_estimate["num_syllables"] * (60 // FRAME_LEN_SEC)
        speech_timestamps, speech_activity = vad_power_thresholding(power, threshold_db=noise_power+10, hangover=20)

    # Onset
    hard_onsets = get_hard_onsets(speech_timestamps, power, threshold=4)

    # Phonation intervals
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

    if not continue_recording:
        return (None, pyaudio.paComplete)
    return (in_data, pyaudio.paContinue)

def update_plot():
    global fig, line1, line2, line3, line2_peaks, line2_vertical_lines, line2_activity, line3_vertical_lines, line4, ax1, ax2, ax3
    global smoothed_rate_list, speech_rate_estimate, zcr, hard_onsets, phonation_intervals
    
    line1.set_ydata(smoothed_rate_list)
        
    for line in line2_vertical_lines:
        line.remove()
    line2_vertical_lines = []
    line2.set_ydata(power)
    line2_activity.set_ydata(speech_activity * 80 - 80)
    line2_peaks.set_data(speech_rate_estimate["peaks"], power[speech_rate_estimate["peaks"]])
    for onset in hard_onsets:
        line2_vertical_lines.append(ax2.axvline(x=onset, color='r'))

    line3.set_ydata(zcr)
    for line in line3_vertical_lines:
        line.remove()
    line3_vertical_lines = []
    for interval in phonation_intervals:
        if (interval[1] - interval[0]) < 20:
            line3_vertical_lines.append(ax3.axvspan(interval[0], interval[1], color='r', alpha=0.2))
        else:
            line3_vertical_lines.append(ax3.axvspan(interval[0], interval[1], color='g', alpha=0.2))

    line4.set_ydata(respiration_filtered)
    line4_activity.set_ydata(speech_activity * 2 - 1)

    fig.canvas.draw()
    fig.canvas.flush_events()

def main():
    global continue_recording

    init_globals()

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_process)

    while(continue_recording):
        update_plot()
        
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    plt.close()
    stop_thread.join()
    if imu_present:
        serial_thread.join()
    imu_respiration_cleanup()

if __name__ == "__main__":
    main()