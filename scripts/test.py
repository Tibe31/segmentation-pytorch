import cv2
import numpy as np


line = np.ones(shape=(20, 475), dtype=np.uint8) * 255

flipped = cv2.rotate(line, cv2.ROTATE_90_CLOCKWISE)

h, w = flipped.shape

radius = int(h / (2 * np.pi))

new_image = np.zeros(shape=(h, radius + w), dtype=np.uint8)
h2, w2 = new_image.shape

new_image[:, w2 - w : w2] = flipped

h, w = new_image.shape
center = (600, 600)

maxRadius = 900

output = cv2.warpPolar(
    new_image,
    center=center,
    maxRadius=radius,
    dsize=(1500, 1500),
    flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR,
)
pass
