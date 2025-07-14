import matplotlib.pyplot as plt
import numpy as np
import matplotlib

class PlotData:
    def __init__(self,
                 rate_list=None,
                 power=None,
                 speech_activity=None,
                 speech_rate_estimate=None,
                 zcr=None,
                 zcr_threshold=None,
                 hard_onsets=None,
                 phonation_intervals=None,
                 respiration_filtered=None):
        self.rate_list = rate_list
        self.power = power
        self.speech_activity = speech_activity
        self.speech_rate_estimate = speech_rate_estimate
        self.zcr = zcr
        self.zcr_threshold = zcr_threshold
        self.hard_onsets = hard_onsets
        self.phonation_intervals = phonation_intervals
        self.respiration_filtered = respiration_filtered

from speech_warnings import speech_rate_warning, phonation_warning

class Plot():
    def __init__(self, dummy_data: PlotData):
        matplotlib.use('TkAgg')

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(20, 16))
        self.line1, = self.ax1.plot(dummy_data.power, color="C0")
        self.line1_activity, = self.ax1.step(np.arange(len(dummy_data.speech_activity)), dummy_data.speech_activity, color='k')
        self.line1_peaks, = self.ax1.plot(dummy_data.speech_rate_estimate["peaks"], dummy_data.power[dummy_data.speech_rate_estimate["peaks"]], 'ro')
        self.line1_vertical_lines = []
        self.line2, = self.ax2.plot(dummy_data.zcr, color="C0")
        self.line2_vertical_lines = []
        self.line3, = self.ax3.plot(dummy_data.respiration_filtered)
        self.ax1.set_ylim(0, 100)
        self.ax1.set_xlim(0, len(dummy_data.power))
        self.ax2.set_ylim(0, 0.4)
        self.fig.suptitle(f'Speech rate: {dummy_data.rate_list[-1]} syllables/min')
        self.ax1.set_title('Signal power (dB)')
        self.ax2.set_title('Zero crossing rate (voicedness)')
        self.ax2.axhline(y=dummy_data.zcr_threshold, color="black", linestyle='--')
        self.ax2.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Power (dB)')
        self.ax2.set_ylabel('ZCR')
        self.ax3.set_title('Respiration')
        self.ax3.set_ylim(-1, 1)
        self.ax3.set_xlim(0, len(dummy_data.respiration_filtered))

        self.text1 = self.fig.text(0.2, 0.025, "SPEECH RATE", fontsize=16, color='gray', fontweight='bold')
        self.text2 = self.fig.text(0.5, 0.025, "PHONATION", fontsize=16, color='gray', fontweight='bold')
        self.text3 = self.fig.text(0.8, 0.025, "BREATHING", fontsize=16, color='gray', fontweight='bold')

        plt.show(block=False)

    def update(self, data: PlotData):
        self.fig.suptitle(f'Speech rate: {data.rate_list[-1]:.1f} syllables/min')

        for line in self.line1_vertical_lines:
            line.remove()
        self.line1_vertical_lines = []
        self.line1.set_ydata(data.power)
        ax1_ylim = self.ax1.get_ylim()
        self.line1_activity.set_ydata(data.speech_activity * (0.99 * np.abs(ax1_ylim[1] - ax1_ylim[0])) + ax1_ylim[0])
        self.line1_peaks.set_data(data.speech_rate_estimate["peaks"], data.power[data.speech_rate_estimate["peaks"]])
        for onset in data.hard_onsets:
            self.line1_vertical_lines.append(self.ax1.axvline(x=onset, color='r'))

        self.line2.set_ydata(data.zcr)
        for line in self.line2_vertical_lines:
            line.remove()
        self.line2_vertical_lines = []
        for interval in data.phonation_intervals:
            if (interval[1] - interval[0]) < 20:
                self.line2_vertical_lines.append(self.ax2.axvspan(interval[0], interval[1], color='r', alpha=0.2))
            else:
                self.line2_vertical_lines.append(self.ax2.axvspan(interval[0], interval[1], color='g', alpha=0.2))

        self.line3.set_ydata(data.respiration_filtered)

        if (speech_rate_warning(data, threshold_rate=150, threshold_frames=3)):
            self.text1.set_color('red')
        else:
            self.text1.set_color('gray')

        if (phonation_warning(data, threshold_duration=20, threshold_num=2)):
            self.text2.set_color('red')
        else:
            self.text2.set_color('gray')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)
