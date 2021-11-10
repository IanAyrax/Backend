from joblib import load
import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
from scipy import stats

def classify(img_eye) :
    radius, center = preprocess(img_eye)
    result = extractFeature(img_eye, radius, center)
    print(result)
    #return str(result)
    return predict(result)

def preprocess(img_eye) :
    #Hough Transform to detect the iris
    real_img = img_eye.copy()
    preprocessing = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    preprocessing = cv2.medianBlur(preprocessing, 7)
    ret, line = cv2.threshold(preprocessing, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    preprocessing = cv2.Canny(preprocessing, 20, 68, apertureSize=3)
    #preprocessing = cv2.Canny(line,0,0)

    rows = preprocessing.shape[0]
    circles = cv2.HoughCircles(preprocessing, cv2.HOUGH_GRADIENT, 1, rows / 4, param1=ret, param2=10, minRadius=250, maxRadius=real_img.shape[0])

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

    z = np.abs(stats.zscore(x))
    x_o = np.array(x)[(z < 3).all(axis=1)]

    clf = KMeans(n_clusters=1, random_state=0).fit(x_o)
    cluster_center = (int(clf.cluster_centers_[0][0]) ,int(clf.cluster_centers_[0][1]))

    cv2.circle(preprocessing, cluster_center, 10, (0, 255, 255), 3)
    cv2.circle(real_img, cluster_center, 10, (0, 255, 255), 3)

    radius = int(real_img.shape[0]/2)
    cv2.circle(preprocessing, cluster_center, radius - 20, (255, 0, 255), 3)
    cv2.circle(real_img, cluster_center, radius - 20, (255, 0, 255), 3)

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
    r_in = int(radius * 0.7)
    r_out = radius

    h = int(radius * 0.3)
    w = int(radius * 2 * np.pi)
    fin_img = rubberSheetModel(img, h, w, r_in, r_out, center)

    #Preprocess Image
    fin_img = cv2.cvtColor(fin_img, cv2.COLOR_BGR2GRAY)

    otsu_threshold, image_result = cv2.threshold(fin_img, 0, 255, cv2.THRESH_OTSU)
    histr = cv2.calcHist([image_result],[0],None,[256],[0,256])

    return otsu_threshold


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