import numpy as np

class VAD:
    def process(self, signal) -> tuple:
        raise NotImplementedError("This method should be overridden by subclasses")

class VADPowerThreshold(VAD):
    def __init__(self, threshold_db=-60, min_speech_duration=5, padding=20):
        super().__init__()
        self.threshold_db = threshold_db
        self.min_speech_duration = min_speech_duration
        self.padding = padding

    def process(self, signal_power_db):
        speech_timestamps = []
        power_thresholded = np.array(signal_power_db >= self.threshold_db).astype(int)

        # Remove short segments
        for i in range(len(power_thresholded)):
            if power_thresholded[i] == 1:
                start = i
                while i < len(power_thresholded) and power_thresholded[i] == 1:
                    i += 1
                end = i - 1
                if (end - start + 1) < self.min_speech_duration:
                    power_thresholded[start:end + 1] = 0

        # Merge adjacent segments
        for i in range(0, len(power_thresholded), self.padding):
            if i + self.padding < len(power_thresholded):
                segment = power_thresholded[i:i + self.padding]
                if np.any(segment == 1):
                    power_thresholded[i:i + self.padding] = 1

        start = -1
        end = -1
        for i in range(0, len(power_thresholded)):
            if (power_thresholded[i] == 1 and i == 0):
                start = i
            elif i != 0 and (power_thresholded[i] == 1 and power_thresholded[i - 1] == 0):
                start = i
            
            if i != 0 and ((power_thresholded[i] == 0 and power_thresholded[i - 1] == 1) 
                        or (power_thresholded[i] == 1 and i == len(power_thresholded) - 1)):
                end = i
                speech_timestamps.append({"start": start, "end": end})
        return speech_timestamps, power_thresholded
    
    def set_threshold(self, threshold_db):
        self.threshold_db = threshold_db
