import numpy as np
import matplotlib.pyplot as plt
import cv2


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def glasses(_img):
    face_cascade = cv2.CascadeClassifier(r'./haarcascades/haarcascade_frontalface_alt.xml')
    eyes_cascade = cv2.CascadeClassifier(r'./haarcascades/haarcascade_eye.xml')

    # read both the images of the face and the glasses
    # image = cv2.imread(r'./images/sample_meduim.jpg')
    glass_img = cv2.imread(r'./images/transparant.png')


    

    img = cv2.imread(_img)

    img = ResizeWithAspectRatio(img, width=1280) # Resize by width OR
    # resize = ResizeWithAspectRatio(image, height=1280) # Resize by height 

    img1 = img.copy()
    print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)[0]
    f_x, f_y, f_w, f_h = face
    print('face', face)
    cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 255, 255), 3)

    l_eye = eyes_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)[0]
    r_eye = eyes_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)[1]


    print ('l_eye', l_eye)
    print ('r_eye', r_eye)

    l_eye_x, l_eye_y, l_eye_w, l_eye_h = l_eye
    r_eye_x, r_eye_y, r_eye_w, r_eye_h = r_eye
    img = cv2.rectangle(img, (l_eye_x + 15, l_eye_y), (l_eye_x + 15 + (l_eye_w + r_eye_w), r_eye_y + r_eye_h), (255,255,255), 5)
    plt.imshow(img)
    """cv2.imshow('frame', img)
    cv2.waitKey()"""

    print(glass_img.shape)
    glasses=cv2.resize(glass_img, ((l_eye_w + r_eye_w) + 25, l_eye_h))
    print(glasses.shape)


    print(glasses[0, 0][2])
    for i in range(glasses.shape[0]):
        for j in range(glasses.shape[1]):
            if (glasses[i, j][2] == 0):
                img1[l_eye_y + i, l_eye_x + j] = glasses[i, j]


    cv2.imshow('frame', img1)
    cv2.waitKey()
    cv2.imwrite(f'glasses_{_img}', img1)


if __name__ == '__main__':

    """ 
    Hermione1
    sample_small
    sample_meduim
    sample_big
    """

    image = r'./images/sample_meduim.jpg'
    glasses(image)