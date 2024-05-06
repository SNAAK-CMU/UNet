import cv2 
from skimage.metrics import mean_squared_error,peak_signal_noise_ratio,structural_similarity


img_path = "test_images/Thailand Project/white_white/img_01063.jpg"
image = cv2.imread(img_path)
(H, W) = image.shape[:2]
# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur the image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Perform the canny operator
canny = cv2.Canny(blurred, 30, 150)

cv2.imshow("canny", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
