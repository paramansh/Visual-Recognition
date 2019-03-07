import cv2
import sys
imname = sys.argv[1]

img = cv2.imread(imname, -1)
# cv2.imshow("image", img)
x = 220
y=80
h=200
w=200
crop_img = img[y:y+h, x:x+w]
cv2.imwrite(imname+"_cropped", crop_img)
# cv2.imshow("croppped", crop_img)
# k=cv2.waitKey(0)
# if k == 27:
    # cv2.destroyAllWindows()
