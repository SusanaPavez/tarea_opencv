import cv2 
import sys

# básicos: el script en py, luego la imagen, y el XML

# image = cv2.imread("bla.jpeg")
# cv2.imshow("Faces found", image)
# cv2.waitKey(0)


# Get user supplied values
imagePath = "caras.png" # esto define el archivo para ver esto 
cascPath = "haarcascade_frontalface_default.xml" # esto define el XML 

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=1,
    minSize=(50, 50),
    # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)