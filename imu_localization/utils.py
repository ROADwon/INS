import numpy as np

def quaternion_multiply(q, r):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def rotate_vector_by_quaternion(v, q):
    v_q = np.array([0, v[0], v[1], v[2]])
    q_inv = quaternion_conjugate(q) / np.dot(q, q)
    rotated_v_q = quaternion_multiply(quaternion_multiply(q, v_q), q_inv)
    return rotated_v_q[1:]