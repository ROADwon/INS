import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from imu_localization.ekf import EKF
from imu_localization.imu_data_handler import IMUDataHandler
from imu_localization.utils import rotate_vector_by_quaternion

# 중력 상수 정의
GRAVITY = 9.80665

# 저역통과 필터 적용 함수
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff=5, fs=1000, order=3):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

# CSV 데이터 로드
data = pd.read_csv("../data/acc_ang.csv")
acc_data = data[["SD_IMU_ACCX", "SD_IMU_ACCY", "SD_IMU_ACCZ"]].values
gyro_data = data[["SD_IMU_P", "SD_IMU_Q", "SD_IMU_R"]].values

# EKF 초기화
state_dim = 6  # x, y, z, roll, pitch, yaw
control_dim = 3  # acc_x, acc_y, acc_z
measurement_dim = 3  # roll, pitch, yaw
freq = 1000  # Hz
dt = 1.0 / freq  # delta t
ekf = EKF(state_dim, control_dim, measurement_dim, dt)

# IMU 데이터 핸들러 초기화
imu_handler = IMUDataHandler(alpha=0.98)

num_samples = len(data)

# 경로 시각화를 위한 위치 데이터 기록
position_history = []
roll_history = []
pitch_history = []
yaw_history = []

# 초기 속도 및 위치
vx, vy, vz = 0.0, 0.0, 0.0
x, y, z = 0.0, 0.0, 0.0

# 가속도 데이터 필터링
acc_data_filtered = lowpass_filter(acc_data, cutoff=5, fs=freq)

# 초기 위치 설정 및 속도 드리프트 보정
initial_velocity = np.array([0.0, 0.0, 0.0])

for i in range(num_samples):
    acc = acc_data_filtered[i]
    gyro = gyro_data[i]

    # 중력 제거
    acc = acc - np.array([0, 0, GRAVITY])

    # IMU 데이터 처리 (Roll, Pitch, Yaw 계산)
    roll, pitch, yaw = imu_handler.process_imu_data(acc, gyro, dt, i)
    deg_roll = math.degrees(roll)
    deg_pitch = math.degrees(pitch)
    deg_yaw = math.degrees(yaw)

    # 쿼터니언을 사용한 회전 변환
    qw = np.cos(yaw / 2) * np.cos(pitch / 2) * np.cos(roll / 2) + np.sin(yaw / 2) * np.sin(pitch / 2) * np.sin(roll / 2)
    qx = np.sin(yaw / 2) * np.cos(pitch / 2) * np.cos(roll / 2) - np.cos(yaw / 2) * np.sin(pitch / 2) * np.sin(roll / 2)
    qy = np.cos(yaw / 2) * np.sin(pitch / 2) * np.cos(roll / 2) + np.sin(yaw / 2) * np.cos(pitch / 2) * np.sin(roll / 2)
    qz = np.cos(yaw / 2) * np.cos(pitch / 2) * np.sin(roll / 2) - np.sin(yaw / 2) * np.sin(pitch / 2) * np.cos(roll / 2)
    quaternion = [qw, qx, qy, qz]

    # 로컬 가속도에서 글로벌 가속도로 변환
    global_acc = rotate_vector_by_quaternion(acc, quaternion)

    # 속도 계산 (가속도 적분)
    vx += global_acc[0] * dt
    vy += global_acc[1] * dt
    vz += global_acc[2] * dt

    # 드리프트 보정 (속도의 누적 오차를 초기화 시점에서 보정)
    if i % 30 == 0:  # 보정 간격을 더 좁혀 오차를 최소화
        vx *= 0.85  # 속도 드리프트를 더 강하게 보정
        vy *= 0.85
        vz *= 0.85

    # 위치 계산 (속도 적분)
    x += vx * dt
    y += vy * dt
    z += vz * dt

    # EKF 업데이트
    control_input = acc  # 가속도 데이터를 제어 입력으로 사용
    ekf.predict(control_input)

    # 측정 갱신 (roll, pitch, yaw 사용)
    measurement = np.array([roll, pitch, yaw])
    ekf.update(measurement)

    # 상태 추정
    state = ekf.get_state()

    # 위치 기록 (x, y 좌표 기록)
    position_history.append([x, y])

    # Roll, Pitch, Yaw 기록
    roll_history.append(deg_roll)
    pitch_history.append(deg_pitch)
    yaw_history.append(deg_yaw)
    print(f"Position: ({state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}) | "
          f"Orientation: ({state[3]:.2f}, {state[4]:.2f}, {state[5]:.2f})")

# 경로 시각화 (x-y 평면에서의 이동 경로)
position_history = np.array(position_history)
plt.plot(position_history[:, 0], position_history[:, 1], label='Estimated Path (X-Y Plane)', color='b')
plt.title('Estimated Path Visualization (X-Y Plane)')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Roll, Pitch, Yaw 시각화
time = np.arange(num_samples) / freq
#
# plt.plot(time, roll_history, label='Roll', color='r')
# plt.title('Roll Over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Roll (degrees)')
# plt.legend()
# plt.grid(True)
#
# plt.plot(time, pitch_history, label='Pitch', color='g')
# plt.title('Pitch Over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Pitch (degrees)')
# plt.legend()
# plt.grid(True)


plt.plot(time, yaw_history, label='Yaw', color='b')
plt.title('Yaw Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Yaw (degrees)')
plt.legend()
plt.grid(True)

plt.show()
