import cv2

front_face_cascade = cv2.CascadeClassifier("c:\\Users\\ACER\\Downloads\\haarcascade_frontalface_default.xml")

profile_face_cascade = cv2.CascadeClassifier("c:\\Users\\ACER\\Downloads\\lbpcascade_profileface.xml")

# capture video from a camera
cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized
while 1:
    
    # reads frames from a camera
    ret, img = cap.read()
    
    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detects frontal faces of different sizes in the input image
    faces = front_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for(x,y,w,h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) # Cyan for frontal faces
        
    # detects profile faces (left-side) of different sizes in the input image
    profiles = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) 
    for(x,y,w,h) in profiles:
        # To draw a rectangle in a face
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)  # Red for left profiles faces
        
    # Detect right profile faces (flipped image)
    gray_flipped = cv2.flip(gray, 1)  # Horizontal flip
    profiles_right = profile_face_cascade.detectMultiScale(gray_flipped, scaleFactor=1.3, minNeighbors=5)
    img_width = img.shape[1]
    for (x, y, w, h) in profiles_right:
        # Flip the x coordinate back to original image
        x_original = img_width - x - w
        cv2.rectangle(img, (x_original, y), (x_original + w, y + h), (0, 255, 0), 2)  # Green rectangle for right profiles faces

        
    # display an image in a window
    cv2.imshow('Face and Profile Detection',img)
    
    # wait for esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
# close the window
cap.release()

# de-allocate any associated memory usage
cv2.destroyAllWindows()