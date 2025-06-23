import imufusion
import serial
import scipy.signal
import numpy as np

def imu_respiration_init():
    global ser, sos, zi, ahrs
    try:
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    except serial.serialutil.SerialException:
        return False

    ahrs = imufusion.Ahrs()
    sos = scipy.signal.butter(4, 0.1, 'low', output='sos')
    zi = scipy.signal.sosfilt_zi(sos)

    return True

def get_respiration_sample():
    global zi
    line = ser.readline().decode('utf-8').strip()
    try:
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = map(float, line.split(','))
        line_is_valid = True
        ahrs.update_no_magnetometer(np.array([gyro_x, gyro_y, gyro_z]), np.array([acc_x, acc_y, acc_z]), 1/12.5)
        x, y, z = ahrs.earth_acceleration
        y_filtered, zi = scipy.signal.sosfilt(sos, [y], zi=zi)

        return y_filtered[0] * 2
    except ValueError:
        return None

def imu_respiration_cleanup():
    try:
        ser.close()
    except NameError:
        pass