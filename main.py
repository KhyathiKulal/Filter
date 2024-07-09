import cv2
import cvzone

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
overlay = cv2.imread('cool.png', cv2.IMREAD_UNCHANGED)

while True:
    _, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_scale)

    if len(faces) > 0:
        # To Find the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Resize and overlay the filter on the largest face
        overlay_resize = cv2.resize(overlay, (int(w * 1.5), int(h * 1.5)))
        frame = cvzone.overlayPNG(frame, overlay_resize, [x - 45, y - 75])

    cv2.imshow('Filter', frame)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
