import cv2
#trained dataset
trainedDataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#image read
img=cv2.imread('images/2.jpg')
# convert into grayscale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = trainedDataset.detectMultiScale(gray)
print(faces)
for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('harish',img)
cv2.waitKey()