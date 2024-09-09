import numpy as np

class EKF:
    def __init__(self, state_dim, control_dim, measurement_dim, dt):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.measurement_dim = measurement_dim

        self.state = np.zeros(state_dim)  # 초기 상태: [x, y, z, roll, pitch, yaw]
        self.covariance = np.eye(state_dim)  # 초기 오차 공분산 행렬

        self.process_noise = np.eye(state_dim) * 0.01
        self.measurement_noise = np.eye(measurement_dim) * 0.1

        self.F = np.eye(state_dim)  # 상태 전이 행렬
        self.B = np.zeros((state_dim, control_dim))  # 제어 입력 행렬

        # 가속도를 위치로 적분하기 위해 B 행렬 초기화
        self.B[0, 0] = dt  # x 축
        self.B[1, 1] = dt  # y 축
        self.B[2, 2] = dt  # z 축

        self.H = np.zeros((measurement_dim, state_dim))  # 측정 모델 행렬
        self.H[:, 3:6] = np.eye(3)  # roll, pitch, yaw 측정에 대한 행렬 요소

    def predict(self, control_input):
        # 예측 단계
        self.state = np.dot(self.F, self.state) + np.dot(self.B, control_input)
        self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.process_noise

    def update(self, measurement):
        # 상태 벡터에서 Roll, Pitch, Yaw 부분 추출
        state_measurement = np.dot(self.H, self.state)  # 3차원 벡터 (Roll, Pitch, Yaw)

        # 갱신 단계
        innovation = measurement - state_measurement  # 크기가 같은 벡터들 간의 연산
        innovation_covariance = np.dot(np.dot(self.H, self.covariance), self.H.T) + self.measurement_noise
        kalman_gain = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(innovation_covariance))

        self.state = self.state + np.dot(kalman_gain, innovation)
        self.covariance = self.covariance - np.dot(np.dot(kalman_gain, self.H), self.covariance)

    def get_state(self):
        return self.state
