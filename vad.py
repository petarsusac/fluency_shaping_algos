import numpy as np

class VAD:
    def process(self, signal) -> tuple:
        raise NotImplementedError("This method should be overridden by subclasses")

class VADPowerThreshold(VAD):
    def __init__(self, threshold_db=-60, hangover=5, start_idx=25):
        super().__init__()
        self.threshold_db = threshold_db
        self.hangover = hangover
        self.start_idx = start_idx

    def process(self, signal_power_db):
        speech_timestamps = []
        power_thresholded = np.array(signal_power_db >= self.threshold_db).astype(int)
        power_thresholded_hangover = np.zeros(len(power_thresholded))
        for i in range(self.start_idx, len(power_thresholded)):
            if power_thresholded[i - min(i, self.hangover):i].any():
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
    
    def set_threshold(self, threshold_db):
        self.threshold_db = threshold_db
