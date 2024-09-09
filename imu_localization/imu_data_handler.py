import math as m
import numpy as np

class IMUDataHandler:
    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.prev_roll = 0.0
        self.prev_pitch = 0.0
        self.prev_yaw = np.deg2rad(180.0)  # 초기 yaw 값을 라디안으로 설정
        self.filtered_Roll = 0.0
        self.filtered_Pitch = 0.0
        self.filtered_Yaw = np.deg2rad(180.0)  # degrees to radians
        self.positions = []

    def deg2rad(self, deg):
        return deg * np.pi / 180

    def rad2deg(self, rad):
        return rad * 180 / np.pi

    def complimentary_filter(self, acc_angle, gyro_angle, alpha=0.98):
        return alpha * gyro_angle + (1 - alpha) * acc_angle

    def normalized_angle(self, angle):
        # Normalize the angle to be within 0 to 2π radians (equivalent to 0 to 360 degrees)
        return angle % (2 * np.pi)

    def process_imu_data(self, acc, gyro, dt, i):
        # 각속도를 라디안으로 변환
        rad = self.deg2rad(gyro)
        Pr = rad[0]
        Qr = rad[1]
        Rr = rad[2]

        acc_roll = np.arctan2(acc[1], acc[2])
        acc_pitch = np.arctan2(-acc[0], np.sqrt(acc[1] ** 2 + acc[2] ** 2))

        if i == 0:
            delta_roll = Pr
            delta_pitch = Qr
            delta_yaw = Rr

            Roll = delta_roll * dt
            Pitch = delta_pitch * dt
            Yaw = delta_yaw * dt

            self.filtered_Roll = self.complimentary_filter(acc_roll, Roll)
            self.filtered_Pitch = self.complimentary_filter(acc_pitch, Pitch)
            self.filtered_Yaw = Yaw  # Yaw는 보정 없이 각속도로만 계산

        else:
            delta_roll = Pr + Qr * (m.sin(self.filtered_Roll) * m.tan(self.filtered_Pitch)) + Rr * (m.cos(self.filtered_Roll) * m.tan(self.filtered_Pitch))
            delta_pitch = Qr * m.cos(self.filtered_Pitch) - Rr * m.sin(self.filtered_Pitch)
            delta_yaw = Rr

            Roll = self.filtered_Roll + delta_roll * dt
            Pitch = self.filtered_Pitch + delta_pitch * dt
            Yaw = self.filtered_Yaw + delta_yaw * dt

            self.filtered_Roll = self.complimentary_filter(acc_roll, Roll)
            self.filtered_Pitch = self.complimentary_filter(acc_pitch, Pitch)
            self.filtered_Yaw = Yaw  # Yaw는 보정 없이 각속도로만 계산

        # Yaw 값을 0 ~ 2π로 정상화
        self.filtered_Yaw = self.normalized_angle(self.filtered_Yaw)

        return self.filtered_Roll, self.filtered_Pitch, self.filtered_Yaw
