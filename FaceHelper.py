import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('sample-image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.6, 4)

sabzeh = cv2.imread("Sabzeh.png", -1)


if len(faces) != 1: 
    raise ValueError('We can only process images with a face. Not more/less.' + str(len(faces)))

for (x, y, w, h) in faces:
    print(x,y,w,h)
    coverResized = cv2.resize(sabzeh, (w, w))
    alpha_l = coverResized[:, :, 2] / 255.0
    alpha_s = 1.0 - alpha_l
    cv2.imshow('cover', coverResized)
    for c in range(0, 3):

        x1, x2 = x, x + w
        y1, y2 = y-110, y + h-110
        print(y2-y1, x2-x1, coverResized.shape[0:2])
        print(alpha_s * coverResized[:, :, c], alpha_l * img[y1:y2, x1:x2, c])
        img[y1:y2, x1:x2, c] = (alpha_s * coverResized[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
    # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

imS = cv2.resize(img, (800, 800))
cv2.imshow('img', imS)
cv2.waitKey(0)  
cv2.destroyAllWindows()