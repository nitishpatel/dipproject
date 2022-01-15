import cv2
import numpy as np
import pyvirtualcam
from colorama import Fore, Back, Style
import colorama
#Select the data source (Webcam) to acquire
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#Get image size from first frame
ret, frame = cap.read()
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
def bright(img, beta_value ):
    img_bright = cv2.convertScaleAbs(img, beta=beta_value)
    return img_bright
def contrast(frame):
    alpha = 1.5 # Contrast control (1.0-3.0)
    beta = 0 # Brightness control (0-100)

    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return adjusted

#HDR effect
def HDR(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return  hdr

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
colorama.init()
# Print the clear screen code '\033[2J'
print('\033[2J')
print(Fore.WHITE+Back.MAGENTA+Style.BRIGHT+"Welcome to the ezyCam")
print(Fore.WHITE+Back.BLACK)
brightness = int(input(Fore.WHITE+Back.CYAN+Style.BRIGHT+"Enter brightness value: "))
isContrast = input(Fore.WHITE+Back.GREEN+Style.BRIGHT+"Do you want to increase contrast? (y/n)")
isFace = input(Fore.BLACK+Back.YELLOW+Style.BRIGHT+"Do you want to detect faces? (y/n)")
addOverlay = input(Fore.BLACK+Back.LIGHTCYAN_EX+Style.BRIGHT+"Do you want to add overlay? (y/n)")
overlayText = ""
if addOverlay == "y":
    overlayText = input("Enter overlay text: ")


with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
    while True:
        #Get the image of each frame
        ret, frame = cap.read()

        #Apply some effect here
       
        #Change color space
        #Enable alpha channel and order in RGB
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # change frame shape
        # frame = cv2.resize(frame, (1280, 720))
        # frame = frame.reshape((720, 1280))
        frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)
        # BRIGHTNESS
        frame = bright(frame, brightness)

        #CONTRAST
        if isContrast == 'y':
            frame = contrast(frame)

        if isFace == 'y':
            gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                if len(faces) > 1:
                    roi = frame[y:y+h, x:x+w]
                    # applying a gaussian blur over this new rectangle area
                    roi = cv2.GaussianBlur(roi, (23, 23), 30)
                    # impose this blurred image on original image to get final image
                    frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

        # Add overlay Text to frame
        if addOverlay == 'y' and overlayText!="":
            cv2.putText(frame,overlayText,(50, 650), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255),2,cv2.LINE_AA)

            # Add Logo to frame
            logo = cv2.imread('logo.png', -1)
            logo = cv2.resize(logo, (200, 100))
            logo = cv2.cvtColor(logo, code=cv2.COLOR_BGR2RGB)
            frame[0:100, 0:200] = logo

        # smooth the webcam image
        # cv2.blur(frame, (3, 3))

        # Canny edge detection
        # cv2.Canny(frame, 100, 200)

        # frame = cv2.flip(frame, 1)
        
        cam.send(frame)

        #Since the image is no longer displayed on the screen, use the function of pyvirtualcam to wait until the next frame.
        cam.sleep_until_next_frame()

#End processing

cap.release()
# import pyvirtualcam
# import numpy as np

# with pyvirtualcam.Camera(width=1280, height=720, fps=20) as cam:
#     print(f'Using virtual camera: {cam.device}')
#     frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
#     print(f'Frame shape: {frame.shape}')
#     print(f'Frame : {frame}')
#     while True:
#         frame[:] = cam.frames_sent % 255  # grayscale animation
#         cam.send(frame)
#         cam.sleep_until_next_frame()