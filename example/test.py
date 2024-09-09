import numpy as np
import math
import pandas as pd

class IMUPathEstimator:

    def __init__(self, alpha=0.9):
        # 필터 설정
        self.alpha = alpha
        self.filtered_acc_x = 0.0
        self.filtered_acc_y = 0.0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.prev_time = None
        self.vel = 0.0
        self.current_pos = np.array([0.0, 0.0], float)
        self.path = []

        # 새로운 필터 변수 추가
        self.filtered_Roll = 0.0
        self.filtered_Pitch = 0.0
        self.filtered_Yaw = 0.0

    def deg2rad(self, deg):
        return np.deg2rad(deg)

    def complimentary_filter(self, acc_angle, gyro_angle):
        return self.alpha * gyro_angle + (1 - self.alpha) * acc_angle

    def normalized_angle(self, angle):
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
            delta_roll = Pr + Qr * (math.sin(self.filtered_Roll) * math.tan(self.filtered_Pitch)) + Rr * (math.cos(self.filtered_Roll) * math.tan(self.filtered_Pitch))
            delta_pitch = Qr * math.cos(self.filtered_Pitch) - Rr * math.sin(self.filtered_Pitch)
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

    def process_csv_data(self, acc, gyro, dt, i):
        # 기존의 Yaw 업데이트
        self.yaw += gyro[2] * dt  # Z축 각속도 사용

        # 저역 통과 필터 적용
        self.filtered_acc_x = self.alpha * self.filtered_acc_x + (1 - self.alpha) * acc[0]
        self.filtered_acc_y = self.alpha * self.filtered_acc_y + (1 - self.alpha) * acc[1]

        self.vel += self.filtered_acc_x * dt

        delta_x = self.vel * math.cos(self.yaw) * dt
        delta_y = self.vel * math.sin(self.yaw) * dt
        self.current_pos += np.array([delta_x, delta_y])

        self.path.append(self.current_pos.copy())

    def get_path(self):
        return np.array(self.path)

    def save_path(self, filename="imu_estimated_path.csv"):
        np.savetxt(filename, self.path, delimiter=',', header="x,y", comments='')

def main():
    estimator = IMUPathEstimator(alpha=0.9)

    # CSV 파일로부터 데이터 읽기
    data = pd.read_csv("../data/acc_ang.csv")
    data.drop(columns=["Unnamed: 0"], inplace=True)
    acc_data = data[["SD_IMU_ACCX", "SD_IMU_ACCY", "SD_IMU_ACCZ"]].values
    gyro_data = data[["SD_IMU_P", "SD_IMU_Q", "SD_IMU_R"]].values


    dt = 0.001  # 시간 간격 설정

    for i in range(len(acc_data)):
        acc = acc_data[i]
        gyro = gyro_data[i]

        # IMU 데이터를 사용하여 경로 및 각도 추정
        estimator.process_csv_data(acc, gyro, dt, i)
        roll, pitch, yaw = estimator.process_imu_data(acc, gyro, dt, i)
        print(f"Sample {i}: Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f}")

    # 추정된 경로 저장
    estimator.save_path("imu_estimated_path.csv")
    print("Estimated path saved to imu_estimated_path.csv")

if __name__ == "__main__":
    main()
