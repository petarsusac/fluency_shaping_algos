import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import librosa
import matplotlib
import scipy.signal
import silero_vad
import threading
import imufusion
import serial

matplotlib.use('TkAgg')

USE_VAD = False
PROC_FRAME_LEN=1024
PROC_HOP_LEN=256
FRAME_LEN_SEC=5
ZCR_THRESHOLD = 0.1
RATE = 16000
CHUNK = 8000
FORMAT = pyaudio.paFloat32
CHANNELS = 1

if USE_VAD:
    vad = silero_vad.load_silero_vad()

def power_vad(signal_power_db, threshold_db=-60, hangover=5):
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

    if i <= 0:
        # 90% point not found
        return 0
    else:
        right = i
    
    # Search to the left for the 10% point
    while (signal[i] > 0.1 * signal[max_idx]) and (i > 0):
        i -= 1

    if i <= 0:
        return 0
    else:
        left = i

    if right != left:
        return (signal[right] - signal[left]) / (right - left)
    else:
        return np.inf

def stop():
    input("Press Enter to stop:")
    global continue_recording
    continue_recording = False

def read_accelerometer_data_from_serial():
    sos = scipy.signal.butter(4, 0.1, 'low', output='sos')
    zi = scipy.signal.sosfilt_zi(sos)

    while continue_recording:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            try:
                acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = map(float, line.split(','))
                ahrs.update_no_magnetometer(np.array([gyro_x, gyro_y, gyro_z]), np.array([acc_x, acc_y, acc_z]), 1/10)
                x, y, z = ahrs.earth_acceleration
                y_filtered, zi = scipy.signal.sosfilt(sos, [y], zi=zi)

                respiration_filtered.pop(0)
                respiration_filtered.append(y_filtered[0] * -2)
            except ValueError:
                # Skip invalid lines
                pass


def init_globals():
    global continue_recording
    global chunks, chunks_power, chunks_zcr, rate_list, smoothed_rate_list, b, a, speech_rate_estimate, speech_timestamps, speech_activity, power, zcr, hard_onsets, phonation_intervals
    global fig, line1, line2, line3, line2_peaks, line2_vertical_lines, line2_activity, line3_vertical_lines, line4, line4_activity, ax1, ax2, ax3, ax4
    global ser, ahrs, respiration_filtered
    global stop_thread, serial_thread, accel_present

    continue_recording = True
    accel_present = True
    chunks = [np.zeros(CHUNK) for _ in range((FRAME_LEN_SEC * RATE) // CHUNK)]
    chunks_power = [signal_power_db(chunk, frame_length=PROC_FRAME_LEN, hop=PROC_HOP_LEN) for chunk in chunks]
    chunks_zcr = [librosa.feature.zero_crossing_rate(y=chunk, frame_length=PROC_FRAME_LEN, hop_length=PROC_HOP_LEN)[0] for chunk in chunks]
    power = np.concatenate(chunks_power)
    zcr = np.concatenate(chunks_zcr)
    b, a = scipy.signal.butter(4, 0.5, 'low')
    rate_list = [0] * 20
    smoothed_rate_list = [0] * 20
    speech_rate_estimate = speech_rate_estimate_power(power, zcr)
    speech_timestamps = []
    speech_activity = np.zeros(len(power))
    hard_onsets = []
    phonation_intervals = []
    respiration_filtered = [0.0] * FRAME_LEN_SEC * 10
    ahrs = imufusion.Ahrs()

    try:
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    except serial.serialutil.SerialException:
        accel_present = False

    stop_thread = threading.Thread(target=stop)
    stop_thread.start()

    if accel_present:
        serial_thread = threading.Thread(target=read_accelerometer_data_from_serial)
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
    ax4.grid()

    plt.show(block=False)

def audio_process(in_data, frame_count, time_info, status):
    global chunks, rate_list, smoothed_rate_list, b, a, speech_rate_estimate, speech_timestamps, speech_activity, zcr, hard_onsets, phonation_intervals, power, zcr

    chunk = np.frombuffer(in_data, dtype=np.float32)
    chunks.pop(0)
    chunks.append(chunk)
    chunk_no_offset = chunk - np.mean(chunk)
    audio_frame = np.concatenate(chunks)
    
    chunks_power.pop(0)
    chunks_power.append(signal_power_db(chunk_no_offset, frame_length=PROC_FRAME_LEN, hop=PROC_HOP_LEN))
    chunks_zcr.pop(0)
    chunks_zcr.append(librosa.feature.zero_crossing_rate(y=chunk_no_offset, frame_length=PROC_FRAME_LEN, hop_length=PROC_HOP_LEN)[0])

    power = np.concatenate(chunks_power)
    zcr = np.concatenate(chunks_zcr)

    speech_rate_estimate = speech_rate_estimate_power(power, zcr)
    zcr = scipy.signal.filtfilt(b, a, zcr)
    hard_onsets = []
    phonation_intervals=[]

    if USE_VAD:
        speech_timestamps = silero_vad.get_speech_timestamps(audio_frame, vad, min_silence_duration_ms=500, threshold=0.3)
        num_syllables = 0
        speech_duration_s = FRAME_LEN_SEC
        speech_activity = np.zeros(len(power))
        # for timestamp in speech_timestamps:
        #     speech_duration_s += (timestamp['end'] - timestamp['start']) / RATE
        for timestamp in speech_timestamps:
                timestamp['start'] = timestamp['start'] // PROC_HOP_LEN
                timestamp['end'] = timestamp['end'] // PROC_HOP_LEN
                speech_activity[timestamp['start']:timestamp['end']] = 1

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
        speech_timestamps, speech_activity = power_vad(power, threshold_db=-60, hangover=30)

    # Onset
    for timestamp in speech_timestamps:
        if (timestamp["start"] != 1):
            power_start = (timestamp["start"] - RATE//20//PROC_HOP_LEN) # 50 ms padding
            power_end = power_start + RATE//10//PROC_HOP_LEN # 100 ms segment
            if power_end < len(power):
                power_segment = power[power_start:power_end]
                if len(power_segment) > 0:
                    slope = rise_slope(power_segment) # 500 ms segment
                    if slope > 4:
                        hard_onsets.append(timestamp["start"])

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
        if (interval[1] - interval[0]) < 10:
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
    if accel_present:
        serial_thread.join()

if __name__ == "__main__":
    main()