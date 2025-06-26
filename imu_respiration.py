import imufusion
import serial
import scipy.signal
import numpy as np

SERIAL_PORT = '/dev/ttyACM0'

class IMUNotFoundError(Exception):
    """Custom exception for IMU not found errors."""
    pass

class IMURespiration:
    def __init__(self):
        self.ahrs = imufusion.Ahrs()
        self.sos = scipy.signal.butter(4, 0.1, 'low', output='sos')
        self.zi = scipy.signal.sosfilt_zi(self.sos)

        try:
            self.ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
        except serial.serialutil.SerialException:
            raise IMUNotFoundError()

    def get_sample(self):
        line = self.ser.readline().decode('utf-8').strip()
        try:
            acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = map(float, line.split(','))
            self.ahrs.update_no_magnetometer(np.array([gyro_x, gyro_y, gyro_z]), np.array([acc_x, acc_y, acc_z]), 1/12.5)
            x, y, z = self.ahrs.earth_acceleration
            y_filtered, self.zi = scipy.signal.sosfilt(self.sos, [y], zi=self.zi)

            return y_filtered[0] * -2
        except ValueError:
            return None

    def close(self):
        try:
            self.ser.close()
        except NameError:
            pass