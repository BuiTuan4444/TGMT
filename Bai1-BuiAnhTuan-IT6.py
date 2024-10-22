import cv2
import numpy as np

image_path = 'D:/PyThon01/TGMT/xp-15.jpg'
image = cv2.imread(image_path, 0)

if image is None:
    print("Không thể đọc ảnh, vui lòng kiểm tra lại đường dẫn.")
    exit()

negative_image = 255 - image
cv2.imwrite('negative_image.jpg', negative_image)
print("Đã lưu ảnh âm tính dưới tên 'negative_image.jpg'")

alpha = 1.5
beta = 0
contrast_enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
cv2.imwrite('contrast_enhanced.jpg', contrast_enhanced)
print("Đã lưu ảnh tăng độ tương phản dưới tên 'contrast_enhanced.jpg'")

c = 255 / np.log(1 + np.max(image))
log_transformed = c * (np.log(1 + image))
log_transformed = np.array(log_transformed, dtype=np.uint8)
cv2.imwrite('log_transformed.jpg', log_transformed)
print("Đã lưu ảnh log dưới tên 'log_transformed.jpg'")

equalized_image = cv2.equalizeHist(image)
cv2.imwrite('equalized_image.jpg', equalized_image)
print("Đã lưu ảnh cân bằng histogram dưới tên 'equalized_image.jpg'")

cv2.imshow("Original Image", image)
cv2.imshow("Negative Image", negative_image)
cv2.imshow("Contrast Enhanced", contrast_enhanced)
cv2.imshow("Log Transformed", log_transformed)
cv2.imshow("Histogram Equalized", equalized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
