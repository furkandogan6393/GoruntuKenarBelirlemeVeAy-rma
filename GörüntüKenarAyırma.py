import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r'C:\Users\TozLu\Desktop\Yapay Zeka\makineogrenimi\Uygulama\resim1.png', cv2.IMREAD_GRAYSCALE)


vertical_filter = np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]])


filtered_image = cv2.filter2D(image, -1, vertical_filter)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Orijinal Görüntü")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Dikey Kenar Tespiti")
plt.imshow(filtered_image, cmap='gray')

plt.show()
