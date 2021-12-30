from joblib import load
import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib.pyplot as plt

def classify(img_eye) :
    img_eye = crop_image(img_eye)
    radius, center = preprocess(img_eye)
    if radius == None and center == None :
        return {'result': "No Circle Detected", 'threshold': 0}

    result, img_rgb = extractFeature(img_eye, radius, center)
    print(result)
    
    returnJson = {'result': predict(result), 'threshold': result}
    
    if int(result) == 0 :
        returnJson = {'result': predict(img_rgb), 'threshold': img_rgb}    
    
    #return str(result)
    #return result, predict(result)
    return returnJson

def crop_image(img) :
    y1 = int(img.shape[0]/1.6) - 40
    y2 = int(img.shape[0]/1.6) + 25

    x1 = int(img.shape[1]/2) - 40
    x2 = int(img.shape[1]/2) + 35

    #cv2.rectangle(img_test, pt1=(x1,y1), pt2=(x2,y2), color=(255, 0, 255), thickness=3)
    cropped_img = img[y1:y2, x1:x2]

    img_phone = cv2.resize(cropped_img, (491, 491), interpolation = cv2.INTER_AREA)

    return img_phone

def preprocess(img_eye) :
    img_eye =cv2.blur(img_eye, (40, 40))
    img_r, img_g, img_b, _ = cv2.mean(img_eye)
    img_bright = (img_r + img_b + img_g) / 3
    
    canny_1 = 50 
    canny_2 = 60

    if img_bright > 60 :
        print("light") 
    else :
        print("dark")
        img_eye = brightness_settings(img_eye, 50)
        canny_1 = 20 
        canny_2 = 20
        
    #Hough Transform to detect the iris
    real_img = img_eye.copy()
    
    preprocessing = contrast_settings(real_img.copy(), 90)

    preprocessing = cv2.cvtColor(preprocessing, cv2.COLOR_BGR2GRAY)
    #preprocessing = cv2.blur(preprocessing, (40, 40))
    preprocessing = cv2.medianBlur(preprocessing, 7)
    ret, line = cv2.threshold(preprocessing, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    preprocessing = cv2.Canny(preprocessing, canny_1, canny_2, apertureSize=3)
    #preprocessing = cv2.Canny(line,0,0)

    rows = preprocessing.shape[0]
    #circles = cv2.HoughCircles(preprocessing, cv2.HOUGH_GRADIENT, 1, rows / 4, param1=ret, param2=10, minRadius=250, maxRadius=real_img.shape[0])
    circles = cv2.HoughCircles(preprocessing, cv2.HOUGH_GRADIENT, 1, rows / 4, param1=ret, param2=10, minRadius=int(real_img.shape[0]/2), maxRadius=real_img.shape[0] - 150)

    img_center = (int(real_img.shape[0]/2),int(real_img.shape[1] /2))
    img_radius = int(real_img.shape[0])

    x = []
    if circles is not None :
        circles = np.uint16(np.around(circles))
        begin = True
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            x.append(center)
    else :
        return None, None

    z = np.abs(stats.zscore(x))
    #x_o = np.array(x)[(z < 3).all(axis=1)]
    x_o = np.array(x)[(z < 1.3).all(axis=1)]

    clf = KMeans(n_clusters=1, random_state=0).fit(x_o)
    cluster_center = (int(clf.cluster_centers_[0][0]) ,int(clf.cluster_centers_[0][1]))

    #radius = int(real_img.shape[0]/2)
    radius = int(real_img.shape[0]/ 2 * 8 / 10)
    #cv2.circle(preprocessing, cluster_center, radius - 20, (255, 0, 255), 3)
    #cv2.circle(real_img, cluster_center, radius - 20, (255, 0, 255), 3)

    return radius, cluster_center

def rubberSheetModel(image, height, width, r_in, r_out, center):
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    #r_out = r_in + r_out
    # Create empty flatten image
    flat = np.zeros((height,width, 3), np.uint8)
    #circle_x = int(image.shape[0] / 2)
    #circle_y = int(image.shape[1] / 2)
    circle_x = int(center[0])
    circle_y = int(center[1])
    
    for i in range(width):
        for j in range(height):
            theta = thetas[i]  # value of theta coordinate
            r_pro = j / height  # value of r coordinate(normalized)

            # get coordinate of boundaries
            Xi = circle_x + r_in * np.cos(theta)
            Yi = circle_y + r_in * np.sin(theta)
            
            Xo = circle_x + r_out * np.cos(theta)
            Yo = circle_y + r_out * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            Xc = (1 - r_pro) * Xi + r_pro * Xo
            Yc = (1 - r_pro) * Yi + r_pro * Yo
            
            #if(Xc > width):
                #Xc = width - 1
            
            if Xc >= image.shape[0] :
                Xc = image.shape[0] - 1
                
            #if(Yc > height):
            #    Yc = height - 1
            
            if Yc >= image.shape[1] :
                Yc = image.shape[1] - 1
            
            color = image[int(Xc)][int(Yc)]  # color of the pixel

            flat[j][i] = color
    
    return flat

def extractFeature(img, radius, center) :
    #r_in = int(radius * 0.7)
    #r_out = radius
    r_in = int(radius * 0.35)
    r_out = int(radius * 0.4)

    #h = int(radius * 0.3)
    #w = int(radius * 2 * np.pi)
    h = int(r_out - r_in)
    w = int(r_out * 2 * np.pi)
    
    rx_1 = int(center[0] - r_out)
    rx_2 = int(center[0] + r_out)

    ry_1 = int(center[1] - r_out)
    ry_2 = int(center[1] + r_out)
    
    fin_img = rubberSheetModel(img[ry_1:ry_2, rx_1:rx_2], h, w, r_in, r_out, center)
    fin_img = cv2.blur(fin_img, (40, 40))

    img_r, img_g, img_b, _ = cv2.mean(fin_img)
    img_rgb = int((img_r + img_b + img_g) / 3)

    #Preprocess Image
    fin_img = cv2.cvtColor(fin_img, cv2.COLOR_BGR2GRAY)

    otsu_threshold, image_result = cv2.threshold(fin_img, 0, 255, cv2.THRESH_OTSU)
    histr = cv2.calcHist([image_result],[0],None,[256],[0,256])

    # plt.imshow(image_result)
    # plt.show()

    return otsu_threshold, img_rgb


def predict(input = 0) :
    clf = load('result_old.joblib') 
    result = clf.predict([[input]])
    result = result[0]

    if result == 0 :
        return "tinggi"
    elif result == 1:
        return "sedang"
    else :
        return "normal"

def brightness_settings(img, brightness=255) :
    if brightness != 0 :
        if brightness > 0 :
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    else :
        buf = img.copy()
        
    return buf

def contrast_settings(img, contrast=127):
    if contrast != 0 :
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)
    else :
        buf = img.copy()
    
    return buf