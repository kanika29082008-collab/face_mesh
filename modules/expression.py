import numpy as np

def eye_aspect_ratio(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices[:6]]
    vertical = (np.linalg.norm(np.array(p2)-np.array(p6)) +
                np.linalg.norm(np.array(p3)-np.array(p5)))
    horizontal = np.linalg.norm(np.array(p1)-np.array(p4))
    return vertical / (2.0 * horizontal)

def mouth_aspect_ratio(landmarks, mouth_indices):
    top, bottom = landmarks[mouth_indices[0]], landmarks[mouth_indices[1]]
    left, right = landmarks[mouth_indices[2]], landmarks[mouth_indices[3]]
    vertical = np.linalg.norm(np.array(top)-np.array(bottom))
    horizontal = np.linalg.norm(np.array(left)-np.array(right))
    return vertical / horizontal
