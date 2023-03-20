import cv2
import numpy as np
from torchvision import transforms


def get_barycentric(image):
    mu = cv2.moments(image*255, False)
    if mu['m00'] != 0:
        barycentric = np.array([
            int(mu['m10'] / mu['m00']), int(mu['m01'] / mu['m00'])])
    else:
        barycentric = np.array([0, 0])
    return barycentric


data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
