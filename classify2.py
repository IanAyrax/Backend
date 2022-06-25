from joblib import load
import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib.pyplot as plt

def classify(img_eye) :
    # img_eye = crop_image(img_eye)
    # radius, center = preprocess(img_eye)
    # if radius == None and center == None :
    #     return {'result': "No Circle Detected", 'threshold': 0}

    # result, img_rgb = extractFeature(img_eye, radius, center)
    # print(result)
    
    # returnJson = {'result': predict(result), 'threshold': result}
    
    # if int(result) == 0 :
    #     returnJson = {'result': predict(img_rgb), 'threshold': img_rgb}    
    input_img = preprocessing(img_eye)
    x, hough_radiuses, canny_img, hough_img = houghTransform(input_img.copy())

    if len(hough_radiuses) == 0 :
        return {'result' : "No Circle Detected", 'threshold' : 0}

    radius, center, x_o = kmeans_radius(x, hough_radiuses)

    fin_h = int(radius)
    fin_w = int(radius * 2 * np.pi)
    normalized_img = rubberSheetModel(input_img, fin_h, fin_w, fin_h, 0, center)

    otsu_threshold, image_result = featureExtraction(normalized_img)

    returnJson = {'result': predict(otsu_threshold), 'threshold': otsu_threshold}
    
    # if int(result) == 0 :
    if int(otsu_threshold) == 0 :
        returnJson = {'result': predict(img_rgb), 'threshold': otsu_threshold}  

    return returnJson

def crop_image(img) :
    y1 = int(img.shape[0]/1.7) - 30
    y2 = int(img.shape[0]/1.7) + 30

    x1 = int(img.shape[1]/2) - 25
    x2 = int(img.shape[1]/2) + 35

    #cv2.rectangle(img_test, pt1=(x1,y1), pt2=(x2,y2), color=(255, 0, 255), thickness=3)
    cropped_img = img[y1:y2, x1:x2]

    img_phone = cv2.resize(cropped_img, (500, 500), interpolation = cv2.INTER_AREA)

    return img_phone

def preprocessing(img) :
    preprocessing = cv2.resize(img, (491, 491), interpolation = cv2.INTER_AREA)
    
    return preprocessing

#Hough Transform to detect the iris
def calculateBrightness(img) :
    s = 0
    for i in range(img.shape[0]) :
        for j in range(img.shape[1]) :
            color = img[i][j]
            s = s + img[i][j][0] + img[i][j][1] + img[i][j][2] 
    level = s/(3 * img.shape[0] * img.shape[1])
    print(level)
    
    return level

def houghTransform(img) :
    real_img = img.copy()
    level = calculateBrightness(img)

    preprocessing = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)

    preprocessing = cv2.medianBlur(preprocessing, 7)
    ret, line = cv2.threshold(preprocessing, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    preprocessing = cv2.Canny(preprocessing, 20, 30, apertureSize=3)

    rows = preprocessing.shape[0]
    circles = cv2.HoughCircles(preprocessing, cv2.HOUGH_GRADIENT, 1, rows / 4, param1=ret, param2=10, minRadius=250, maxRadius=real_img.shape[0])

    img_center = (int(real_img.shape[0]/2),int(real_img.shape[1] /2))
    img_radius = int(real_img.shape[0])

    z_img = preprocessing.copy()

    x = []
    hough_radiuses = []
    if circles is not None :
        circles = np.uint16(np.around(circles))
        begin = True
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(preprocessing, center, radius, (255, 0, 255), 3)

            x.append(center)

            hough_radiuses.append(radius)
            
    return x, hough_radiuses, z_img, preprocessing

def closest_point(point, points) :
    points = np.asarray(points)
    
    deltas = points - point
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    closest_idx2 = np.argmin(dist_2)
    
    print(f'dist2 : {dist_2}')
    
    return closest_idx2, dist_2[closest_idx2] 

def kmeans_radius(x, hough_radiuses) :
    z = np.abs(stats.zscore(x))
    x_o = np.array(x)[(z < 1.3).all(axis=1)]
    radiuses_o = np.array(hough_radiuses)[(z < 1.3).all(axis=1)]

    clf = KMeans(n_clusters=1, random_state=0).fit(x_o)
    cluster_center = (int(clf.cluster_centers_[0][0]) ,int(clf.cluster_centers_[0][1]))

    closest_idx, closest_distance = closest_point(cluster_center, x_o)

    if closest_distance <= 6000 :
        radius = radiuses_o[closest_idx]
        cluster_center = (x_o[closest_idx][0], x_o[closest_idx][0])

    else :
        radius = int(np.average(hough_radiuses) / 2)

    radius = radius - 30

    return radius, cluster_center, x_o 

def segmentation(img) :
    preprocessing = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    preprocessing = cv2.medianBlur(preprocessing, 5)
    ret, _ = cv2.threshold(preprocessing, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    rows = preprocessing.shape[0]
    circles = cv2.HoughCircles(preprocessing, cv2.HOUGH_GRADIENT, 1, rows / 4, param1=120, param2=10, minRadius=250, maxRadius=300)
    
    if circles is not None :
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(preprocessing, center, radius, (255, 0, 255), 3)
    
    return preprocessing, radius

def removeReflection(img):
    ret, mask = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
    kernel = np.ones((10, 20), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    grayscale = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)

    preprocessing = cv2.inpaint(img, grayscale, 5, cv2.INPAINT_TELEA)
    
    return preprocessing

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

def featureExtraction(fin_img) :
    #Preprocess Image
    fin_img = removeReflection(fin_img)
    brightness = calculateBrightness(fin_img[0:int(fin_img.shape[0]*0.7), 0:int(fin_img.shape[1])])
    # print("Brightness = " + str(brightness))
    fin_img = fin_img[int(fin_img.shape[0]*0.7):fin_img.shape[0], 0:int(fin_img.shape[1])]
    
    fin_img = cv2.cvtColor(fin_img, cv2.COLOR_BGR2GRAY)
    fin_img = cv2.GaussianBlur(fin_img, (7, 7), 0)

    otsu_threshold, image_result = cv2.threshold(fin_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if brightness > 85.5 :
        otsu_threshold = otsu_threshold - 30 
    
    # print("OTSU Threshold = " + str(otsu_threshold))
    
    
    return otsu_threshold, image_result

def predict(input = 0) :
    clf = load('result_new.joblib') 
    result = clf.predict([[input]])
    result = result[0]

    if result == 0 :
        return "tinggi"
    elif result == 1:
        return "sedang"
    else :
        return "normal"
