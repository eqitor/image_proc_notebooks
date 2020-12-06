import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage.draw import line as skimage_line
import math


# Shows single image, prepared to use with %matplotlib notebook
# @img - source image
# @title - title of image
# @cmap - colormap setting, gray as default
def simg(img, title, cmap="gray"):
    plt.figure(title)
    plt.imshow(img, cmap)
    plt.title(title)
    plt.show()
    

# Shows single image
# @img - source image
# @title - title of image
# @scale - scale image to original size if arg == "scale"
def showimage(img, title, scale=""):
    if scale == "scale":
        plt.figure(title, figsize=(img.shape[0]/100, img.shape[1]/100))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.show()
    
    
# Returning blur image
# @img - source image
# @blur_iterations - iterations of GaussianBlur algorithm used on single image
# @kernel_size - size of kernel used in GaussianBlur filter
# @sigma - sigma parameter for GaussianBlur filter
# 
# @return img_gray - returns blurred image
def create_blur_image(img, blur_iterations=1, kernel_size=5, sigma=0):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(0, blur_iterations):
        img_gray = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), sigma)
    return img_gray


# Calculates expected value of given distribution
# @dist - source distribution
#
# @return EX - expected value
def ex(dist):
    integral = np.sum(dist)
    EX = 0
    for x, p in enumerate(dist):
        EX += x*(p[0]/integral)
    return EX


# Removes uneven ilumination of image
# @img - source image
# @blur_iterations - iterations of GaussianBlur algorithm used on single image
# @kernel_size - size of kernel used in GaussianBlur filter
# @sigma - sigma parameter for GaussianBlur filter
# @img_weight - weight of image array used in blending
# @blur_weight - weight of blur image array used in blending
#
# @return img_gray_filtered - filtered image in grayscale
def remove_uneven_ilumination(img, blur_iterations=1, kernel_size=121, sigma=100, img_weight=0.5, blur_weight=0.5):
    if img.ndim == 3:
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # prepare blur filter
    blur = img
    for i in range(0, blur_iterations):
        blur = cv2.GaussianBlur(blur, (kernel_size, kernel_size), sigma)
    
    # removing uneven ilumination
    img_gray_filtered = cv2.addWeighted(img, img_weight, -blur, blur_weight, 0)
    
    return img_gray_filtered
    
    
# Preparing image to lesion detection using Otsu thresholding
# @img - source image
# @kernel_size - size of kernel used in GaussianBlur filter
# @sigma - sigma parameter for GaussianBlur filter
# @blur_iterations - iterations of GaussianBlur algorithm used on single image
# @img_weight - weight of image array used in blending
# @blur_weight - weight of blur image array used in blending
# @median_filter - enable using median filter on output image
# @blur_img - custom blur image for blending
#
# @reutrn img_mediane - returns thresholded image with effect of median filter if @median_filter is True
# @return img_thresholded - returns thresholded image
def prepare_image_otsu(img, kernel_size=5, sigma=0, blur_iterations=1, img_weight=1.1, blur_weight=1.3,
                  median_filter=False, blur_img=[]):
    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # prepare blur filter
    if blur_img != []:
        blur = blur_img
    else:
        blur = img_gray
        for i in range(0, blur_iterations):
            blur = cv2.GaussianBlur(blur, (kernel_size, kernel_size), sigma)
    
    # removing uneven ilumination
    img_gray_filtered = cv2.addWeighted(img_gray, img_weight, -blur, blur_weight, 0)

    # Otsu thresholding
    ret, img_thresholded = cv2.threshold(img_gray_filtered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Using median filter (optional)
    if median_filter:
        img_median = cv2.medianBlur(img_thresholded, 5)
        return img_mediane
    
    return img_thresholded



# Preparing image to lesion detection using mean value of background calculated with Otsu algorithm as point of
# background corrected distribution slicing.
# @img - source image
# @kernel_size - size of kernel used in GaussianBlur filter
# @sigma - sigma parameter for GaussianBlur filter
# @blur_iterations - iterations of GaussianBlur algorithm used on single image
# @img_weight - weight of image array used in blending
# @blur_weight - weight of blur image array used in blending
# @median_filter - enable using median filter on output image
# @blur_img - custom blur image for blending
#
# @reutrn img_mediane - returns thresholded image with effect of median filter if @median_filter is True
# @return img_thresholded - returns thresholded image
def prepare_image_otsu_slicing(img, kernel_size=121, sigma=100, blur_iterations=1, img_weight=0.5, blur_weight=0.5,
                  median_filter=False, blur_img=[]):
    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # prepare blur filter
    if blur_img != []:
        blur = blur_img
    else:
        blur = img_gray
        for i in range(0, blur_iterations):
            blur = cv2.GaussianBlur(blur, (kernel_size, kernel_size), sigma)
    
    
    img_gray_filtered = cv2.addWeighted(img_gray, img_weight, -blur, blur_weight, 0)
    
    # avarage thresholding
    fpb = cv2.calcHist([img_gray_filtered], channels=[0], mask=None, histSize=[256], ranges=[0, 255])
    
    
    
    fpb_norm = fpb.ravel()/fpb.sum()
    Q = fpb_norm.cumsum()

    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1

    for i in range(1, 256):
        p1,p2 = np.hsplit(fpb_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    
    ub_ind = int(m1)
    fpb_right_lobe = fpb[ub_ind:2*ub_ind]
    fpb_right_lobe_rev = np.flip(fpb_right_lobe)
    fb_prime = np.concatenate((fpb_right_lobe_rev, fpb[ub_ind:]),0)
    fp_prime = fpb - fb_prime
    fp_prime = [[x if x>=0 else 0 for x in row] for row in fp_prime]


    ex_fp = ex(fp_prime)
    ex_fb = ex(fb_prime)


    T = int((ex_fp + ex_fb)/2)
    ret, img_thresholded = cv2.threshold(img_gray_filtered, T, 255, cv2.THRESH_BINARY)
    
    
    # Using median filter (optional)
    if median_filter:
        img_median = cv2.medianBlur(img_thresholded, 5)
        return img_mediane
    
    return img_thresholded


# Preparing image to lesion detection using peak of image distribution as point of slicing distribution.
# @img - source image
# @kernel_size - size of kernel used in GaussianBlur filter
# @sigma - sigma parameter for GaussianBlur filter
# @blur_iterations - iterations of GaussianBlur algorithm used on single image
# @img_weight - weight of image array used in blending
# @blur_weight - weight of blur image array used in blending
# @median_filter - enable using median filter on output image
# @blur_img - custom blur image for blending
# @optimise_resolution - if True, sets image resolution to 128x128, False as default
# @testing - if True, showing additional plots of distribution, False as default
#
# @reutrn img_mediane - returns thresholded image with effect of median filter if @median_filter is True
# @return img_thresholded - returns thresholded image
def prepare_image_peak_slicing(img, kernel_size=121, sigma=100, blur_iterations=1, img_weight=0.5, blur_weight=0.5,
                  median_filter=False, blur_img=[], optimise_resolution=False, testing=False):
    
    
    if optimise_resolution:
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    
    if img.ndim == 3:# convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # prepare blur filter
    if blur_img != []:
        blur = blur_img
    else:
        blur = img
        for i in range(0, blur_iterations):
            blur = cv2.GaussianBlur(blur, (kernel_size, kernel_size), sigma)
    
    
    img_gray_filtered = cv2.addWeighted(img, img_weight, -blur, blur_weight, 0)
    
    # avarage thresholding
    fpb = cv2.calcHist([img_gray_filtered], channels=[0], mask=None, histSize=[256], ranges=[0, 255])   
    ub_ind = int(np.argmax(fpb))
    fpb_right_lobe = fpb[ub_ind:2*ub_ind] 
    fpb_right_lobe_ext = np.zeros((ub_ind, 1))
    fpb_right_lobe_ext[:fpb_right_lobe.shape[0], :fpb_right_lobe.shape[1]] = fpb_right_lobe
    fpb_right_lobe_rev = np.flip(fpb_right_lobe_ext)
    fb_prime = np.concatenate((fpb_right_lobe_rev, fpb[ub_ind:]),0)
    fp_prime = fpb - fb_prime
    fp_prime = np.abs(fp_prime)
    fp_prime = [[x if x>=0 else 0 for x in row] for row in fp_prime]
    
    ex_fp = ex(fp_prime)
    ex_fb = ex(fb_prime)

    T = int((ex_fp + ex_fb)/2)
    
    ret, img_thresholded = cv2.threshold(img_gray_filtered, T, 255, cv2.THRESH_BINARY)
    
    if testing:
        plt.plot(fpb)
        plt.xlabel("Wartośc piksela")
        plt.ylabel("Ilość pikseli")
        plt.title("Oryginalny histogram")
        plt.grid()
        plt.show()

        plt.plot(fpb_right_lobe)
        plt.xlabel("Wartośc piksela")
        plt.ylabel("Ilość pikseli")
        plt.title("Prawy płat oryginalnego histogramu")
        plt.grid()
        plt.show()

        plt.plot(fpb_right_lobe_ext)
        plt.xlabel("Wartośc piksela")
        plt.ylabel("Ilość pikseli")
        plt.title("Rozszerzony prawy płat oryginalnego histogramu")
        plt.grid()
        plt.show()

        plt.plot(fpb_right_lobe_rev)
        plt.xlabel("Wartośc piksela")
        plt.ylabel("Ilość pikseli")
        plt.title("Odwrócony prawy płat histogramu")
        plt.grid()
        plt.show()

        plt.plot(fb_prime)
        plt.xlabel("Wartośc piksela")
        plt.ylabel("Ilość pikseli")
        plt.title("Uzyskany histogram tła")
        plt.grid()
        plt.show()

        plt.plot(fp_prime)
        plt.xlabel("Wartośc piksela")
        plt.ylabel("Ilość pikseli")
        plt.title("Uzyskany histogram zniekształcenia dermatologicznego")
        plt.grid()
        plt.show()
        
        print("T = {}".format(T))
    
    
    # Using median filter (optional)
    if median_filter:
        img_median = cv2.medianBlur(img_thresholded, 5)
        return img_median
    
    return img_thresholded



# Preparing image to lesion detection using mean value of background calculated with Otsu algorithm as point of
# background corrected distribution slicing, drawing 3d plot in addition
# @img - source image
# @kernel_size - size of kernel used in GaussianBlur filter
# @sigma - sigma parameter for GaussianBlur filter
# @blur_iterations - iterations of GaussianBlur algorithm used on single image
# @img_weight - weight of image array used in blending
# @blur_weight - weight of blur image array used in blending
# @median_filter - enable using median filter on output image
# @blur_img - custom blur image for blending
#
# @reutrn img_mediane - returns thresholded image with effect of median filter if @median_filter is True
# @return img_thresholded - returns thresholded image
def prepare_image2_otsu_slicing_3dplot(img, kernel_size=5, sigma=0, gamma=0, blur_iterations=1, img_weight=0.5, blur_weight=0.5,
                  median_filter=False, blur_img=[]):
    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # prepare blur filter
    if blur_img != []:
        blur = blur_img
    else:
        blur = img_gray
        for i in range(0, blur_iterations):
            blur = cv2.GaussianBlur(blur, (kernel_size, kernel_size), sigma)
      
    img_gray_filtered = cv2.addWeighted(img_gray, img_weight, -blur, blur_weight, 0)
    
    # avarage thresholding
    fpb = cv2.calcHist([img_gray_filtered], channels=[0], mask=None, histSize=[256], ranges=[0, 255])
    
    fpb_norm = fpb.ravel()/fpb.sum()
    Q = fpb_norm.cumsum()

    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1

    for i in range(1, 256):
        p1,p2 = np.hsplit(fpb_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    
    ub_ind = int(m1)
    fpb_right_lobe = fpb[ub_ind:2*ub_ind]
    fpb_right_lobe_rev = np.flip(fpb_right_lobe)
    fb_prime = np.concatenate((fpb_right_lobe_rev, fpb[ub_ind:]),0)
    fp_prime = fpb - fb_prime
    fp_prime = [[x if x>=0 else 0 for x in row] for row in fp_prime]


    ex_fp = ex(fp_prime)
    ex_fb = ex(fb_prime)



    T = int((ex_fp + ex_fb)/2)
    
    ret, img_thresholded = cv2.threshold(img_gray_filtered, T, 255, cv2.THRESH_BINARY)
    
    ret_o, img_thresholded_o = cv2.threshold(img_gray_filtered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    
    # 3d plotting
    
    T_surface = np.full_like(img_gray_filtered, T)
    ex_fp_surface = np.full_like(img_gray_filtered, ex_fp)
    ex_fb_surface = np.full_like(img_gray_filtered, ex_fb)
    ret_o_surface = np.full_like(img_gray_filtered, ret_o)
    
    ny, nx = img_gray_filtered.shape

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    xv, yv = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    image_gray_3d = ax.plot_surface(xv, yv, img_gray_filtered, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    T_surface_3d = ax.plot_wireframe(xv, yv, T_surface, rstride=20, cstride=20)
    ax.text(0, 0, T, "T", color='blue')
    
    ex_fp_surface_3d = ax.plot_wireframe(xv, yv, ex_fp_surface, rstride=20, cstride=20, color="red")
    ax.text(0, 0, ex_fp, "ex_fp", color='red')
    
    ex_fb_surface_3d = ax.plot_wireframe(xv, yv, ex_fb_surface, rstride=20, cstride=20, color="green")
    ax.text(0, 0, ex_fb, "ex_fb", color='green')
    
    ret_o_3d = ax.plot_wireframe(xv, yv, ret_o_surface, rstride=20, cstride=20, color="orange")
    ax.text(0, 0, ret_o, "Otsu", color='orange')
    
    plt.show()
    
    # Using median filter (optional)
    if median_filter:
        img_median = cv2.medianBlur(img_thresholded, 5)
        return img_mediane
    
    return img_thresholded


# Finds contour of pigmented area on thresholded image.
# @img - source thresholded image
# @return max_length_cnt - list of points representing finded contour
def find_pigmented_contour(img):
    img_blur = cv2.GaussianBlur(img, (7, 7), sigmaX=1)
    img_canny = cv2.Canny(img_blur, 100, 200)
    img_dilation = cv2.dilate(img_canny, np.ones((3, 3), np.uint8), iterations=1)
    
    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_length = 0
    max_length_cnt = None
    
    for cnt in contours:
        if cv2.arcLength(cnt,True) > max_length:
            max_length = cv2.arcLength(cnt, True)
            max_length_cnt = cnt

    
    for cnt in contours:
        if cv2.arcLength(cnt, True) > max_length:
            max_length = cv2.arcLength(cnt, True)
            max_length_cnt = cnt
    
    return max_length_cnt

# Drawing connected points on image.
# @dest - image to drawn
# @cnt - list of points to connect
# @color - tuple representing color in BGR format
# @thickness - thickness of the drawn line
def connect_points(dest, cnt, color, thickness):
    for i in range(1, cnt.shape[0]):
        cv2.line(dest, (cnt[i][0][0], cnt[i][0][1]), (cnt[i-1][0][0], cnt[i-1][0][1]), color, thickness)
        
# Creates grayscale image with connected points
# @img - source image
# @cnt - contour to drawn
#
# @return converted - new image with contour
def shape_matrix(img, cnt):
    
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    temp_img = np.full_like(img, 255)
    connect_points(temp_img, cnt, (0, 0, 0), 1)
#     converted = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
    return temp_img


# Finds bounding box of contour.
# @cnt - source contour
#
# @return x, y, w, h - x,y - point of right upper corner; w,h - width and height
def find_bounding_box(cnt):
    return cv2.boundingRect(cnt)


# Finds convex hull of contour.
# @cnt - source contour
#
# @return ... - x,y - points representing convex hull of contour
def find_convex_hull(cnt):
    return cv2.convexHull(cnt)

# Finds normal axe of given axe in between contour points.
# @cnt - source contour
# @pt1 - start point of given axe
# @pt2 - end point of given axe
# @tolerance - tolerance of acceptable mismatch between new axe and contour points.
#
# @return pt1_res, pt2_res - start point and end point of new finded axe
def find_normal_axe(cnt, pt1, pt2, tolerance=2):
    # Calculate S - point in the middle between pt1 and pt2
    S = (int((pt1[0][0]+pt2[0][0])/2), int((pt1[0][1]+pt2[0][1])/2))
    
    # Calculate a_1
    a_1 = (pt2[0][1] - pt1[0][1])/(pt2[0][0] - pt1[0][0])
    
    # Calculate a_2
    a_2 = -1/a_1
    
    # Calculate b
    b = S[1] - a_2*S[0]
    
    # Calculate max_x
    max_x = max(cnt[:,:,0])[0]
    
    pt1_res = None
    pt2_res = None
    
    temp_tol = tolerance
    
    while pt1_res is None:
        for x in range(S[0], max_x):
            y = int(a_2*x + b)
            for x_cnt, y_cnt in cnt[:,0,:]:
                if x_cnt-temp_tol < x < x_cnt+temp_tol and y_cnt-temp_tol < y < y_cnt+temp_tol:
                    pt1_res = (x, y)
                    break
        if pt1_res is None:
            temp_tol += 3
            
    temp_tol = tolerance
            
    while pt2_res is None:
        for x in range(0, S[0]-1):
            y = int(a_2*x + b)
            for x_cnt, y_cnt in cnt[:,0,:]:
                if x_cnt-temp_tol < x < x_cnt+temp_tol and y_cnt-temp_tol < y < y_cnt+temp_tol:
                    pt2_res = (x, y)
                    break
        if pt2_res is None:
            temp_tol += 3
    
    return pt1_res, pt2_res


# Finds axes of pigmented area.
# @cnt - contour of pigmented area
# @tolerance - tolerance of acceptable mismatch between normal axe and contour points.

# @return ... - points representing axes 
def find_axes(cnt, tolerance=2):
    
    longest_axe = 0
    pt1_long = None
    pt2_long = None
    for pt1 in cnt:
        for pt2 in cnt:
            lenght = int(math.sqrt((pt2[0][0] - pt1[0][0])**2 + (pt2[0][1] - pt1[0][1])**2))
            if lenght > longest_axe:
                longest_axe = lenght
                pt1_long = pt1
                pt2_long = pt2
                
    pt1_norm, pt2_norm = find_normal_axe(cnt, pt1_long, pt2_long, tolerance)
                    
    return (pt1_long[0][0], pt1_long[0][1]), (pt2_long[0][0], pt2_long[0][1]), pt1_norm, pt2_norm
    
    
# Extends contour with additional points between points of given contours (like they are connected with lines).
# @cnt - source contour
#
# @return ... - new extended contour
def repair_contour(cnt):
    new_contour = []
    mod = cnt.shape[0]
    for i in range(1, cnt.shape[0]+1):
        ind = i%mod
        line_x, line_y = skimage_line(cnt[ind][0][0], cnt[ind][0][1], cnt[ind-1][0][0], cnt[ind-1][0][1])
        for x, y in zip(line_x, line_y):
            new_contour.append([[x, y]])
    
    return np.array(new_contour)
    
    
# Calculate A_p - area of pigmented contour
# @cnt - source contour
#
# @return ... - area of pigmented contour
def calc_A_p(cnt):
    return int(cv2.contourArea(cnt))


# Calculate A_c - area of convex hull.
# @hull - source convex hull
#
# @return ... - area of convex hull
def calc_A_c(hull):
    repaired_hull = repair_contour(hull)
    return int(cv2.contourArea(repaired_hull))


# Calculate A_b - area of bounding box.
# @cnt - source contour
#
# @return ... - area of bounding box
def calc_A_b(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return int(w*h)

# Calculate distance between two given points.
# @pt1 - first point
# @pt2 - second point
#
# @return ... - distance between point pt1 and pt2
def calc_length(pt1, pt2):
    return int(math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2))


# Calculate p_p - perimeter of contour.
# @cnt - source contour
#
# @return ... - perimeter of contour
def calc_p_p(cnt):
    return int(cv2.arcLength(cnt, False))


# Calculate a_b and b_b parameters of contour.
# @cnt - source contour
#
# @return a_b, b_b - sizes of bounding rectangle
def calc_a_b_b_b(cnt):
    x, y, a_b, b_b = cv2.boundingRect(cnt)
    return a_b, b_b



# Calculate entropy of given image (prepared image)
# @prepared_image - source image (prepared with preparing function)
#
# @return entropy - computed entropy of image
def entropy(prepared_image):
    sum_of_black = 0
    sum_of_white = 0
    
    for row in prepared_image:
        for pixel in row:
            if pixel == 0:
                sum_of_black += 1
            elif pixel == 255:
                sum_of_white += 1
    p_white = sum_of_white/(sum_of_black+sum_of_white)
    p_black = sum_of_black/(sum_of_black+sum_of_white)
    entropy = -(p_white*np.log2(p_white) + p_black*np.log2(p_black))
    return entropy



# Computes assymetry parameters of image
# @img - source image
#
# @return ... - computed parameters
def assymetry_quantification(img):
    
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img = prepare_image_peak_slicing(img, median_filter=True)
    
    contour = find_pigmented_contour(img)
    
    bounding_box = find_bounding_box(contour)
    
    convex_hull = find_convex_hull(contour)
    
    repaired_contour = repair_contour(contour)
    
    a_p_0, a_p_1, b_p_0, b_p_1 = find_axes(repaired_contour)
    
    A_p = calc_A_p(repaired_contour)
    
    A_c = calc_A_c(convex_hull)
    
    A_b = calc_A_b(repaired_contour)
    
    p_p = calc_p_p(repaired_contour)
    
    a_p = calc_length(a_p_0, a_p_1)
    
    b_p = calc_length(b_p_0, b_p_1)
    
    a_b, b_b = calc_a_b_b_b(repaired_contour)
    
    entr = entropy(img)
    
    
    features = {
        'A_p' : A_p,
        'A_c' : A_c,
        'solidity' : A_p/A_c,
        'extent' : A_p/A_b,
        'equivalent diameter' : (4*A_p)/np.pi,
        'circularity' : (4*np.pi*A_p)/(p_p**2),
        'p_p' : p_p,
        'b_p/a_p' : b_p/a_p,
        'b_b/a_b' : b_b/a_b,
        'entropy' : entr,
        
    }
    
    return features
    
    
    
    
    


# Computes colour parameters of image
# @img - source image
#
# @return u, o2 - computed parameters      
def border_quantification(img):
    
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img = remove_uneven_ilumination(img)
    
    img_gray_sobel_edgesx = cv2.Sobel(img, -1, dx=1, dy=0, scale=1, delta=0,
                                  borderType=cv2.BORDER_DEFAULT)
    img_gray_sobel_edgesy = cv2.Sobel(img, -1, dx=0, dy=1, scale=1, delta=0,
                                  borderType=cv2.BORDER_DEFAULT)
    img_gray_sobel_edges = img_gray_sobel_edgesx + img_gray_sobel_edgesy
    
    img_thres = prepare_image_peak_slicing(img, median_filter=True)
    
    contour = find_pigmented_contour(img_thres)
    contour_repaired = repair_contour(contour)
    contour_repaired_img = shape_matrix(img, contour_repaired)
    contour_repaired_img_dilate = cv2.bitwise_not(contour_repaired_img)
    contour_repaired_img_dilate = cv2.dilate(contour_repaired_img_dilate, np.ones((3, 3), np.uint8), iterations=1)
    
    
    
    # Calculate u
    u_sum = 0
    N = 0 # to count in loop
    for x in range(0, contour_repaired_img_dilate.shape[0]):
        for y in range(0, contour_repaired_img_dilate.shape[1]):
            if contour_repaired_img_dilate[x][y] != 0:
                N+=1
                u_sum += abs(img[x][y]*img_gray_sobel_edges[x][y])

    u = u_sum/N
    
    # Calculate o
    o2_sum = 0
    for x in range(0, contour_repaired_img_dilate.shape[0]):
        for y in range(0, contour_repaired_img_dilate.shape[1]):
            if contour_repaired_img_dilate[x][y] != 0:
                o2_sum += (abs(img[x][y]*img_gray_sobel_edges[x][y]) - u)**2

    o2 = o2_sum/N
    
    features = {
        'u' : u,
        'o2' : o2,
    }
    
    return features



# Creates contour mask using "flood fill" algorithm
# @img - source image
# @cnt - contour to fill
#
# @return img_shape - created mask as image
def create_contour_mask(img, cnt):
    img_shape = shape_matrix(img, cnt)
    img_shape = cv2.bitwise_not(img_shape)
    img_shape = cv2.dilate(img_shape, np.ones((2, 2), np.uint8), iterations=1)
    img_shape = cv2.bitwise_not(img_shape)
    result = cv2.floodFill(img_shape, None,(1, 1), 0)
    return img_shape

    
# Computes Euclidian distance between two points in three-dimentional space
# @pt1 - first point
# @pt2 - second point
#
# @return .. - computed distance
def bgr_euclidian_distance(pt1, pt2):
    return int(math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2))


# Computes colour parameters of image
# @img - source image
#
# @return color_counter, B_params, G_params, R_params - computed parameters
def colour_quantification(img):
    
    img_prepared = prepare_image_peak_slicing(img, median_filter=True)
    contour = find_pigmented_contour(img_prepared)
    repaired_contour = repair_contour(contour)
    
    mask = create_contour_mask(img, repaired_contour)
    
    img_blur = cv2.GaussianBlur(img, (5, 5), 2)
    
    # BGR FORMAT
    basic_colors = {
        'WHITE' : (255, 255, 255),
        'RED' : (51, 51, 204),
        'LIGHT_BROWN' : (0, 102, 153),
        'DARK_BROWN' : (0, 0, 51),
        'BLUE_GRAY' : (255, 153, 51),
        'BLACK' : (0, 0, 0),
    }
    
    # counters
    color_counter = {
        'WHITE' : 0,
        'RED' : 0,
        'LIGHT_BROWN' : 0,
        'DARK_BROWN' : 0,
        'BLUE_GRAY' : 0,
        'BLACK' : 0,
    }
    
    # Number of pixels in pigmented area
    n = int(np.sum(mask[mask == 255])/255)
    
    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            min_dist = np.inf
            min_dist_color = None
            if mask[x][y] == 255:
                for color, bgr_val in basic_colors.items():
                    dist = bgr_euclidian_distance(img[x][y], bgr_val)
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_color = color
                color_counter[min_dist_color] += 1/n
            
            
    B, G, R = cv2.split(img)

    B = B.flatten()
    G = G.flatten()
    R = R.flatten()
    
    
    features = {
        'WHITE' : color_counter['WHITE'],
        'RED' : color_counter['RED'],
        'LIGHT_BROWN' : color_counter['LIGHT_BROWN'],
        'DARK_BROWN' : color_counter['DARK_BROWN'],
        'BLUE_GRAY' : color_counter['BLUE_GRAY'],
        'BLACK' : color_counter['BLACK'],
        'B_mean' : np.mean(B),
        'B_variance' : np.var(B),
        'B_min' : min(B),
        'B_max' : max(B),
        'G_mean' : np.mean(G),
        'G_variance' : np.var(G),
        'G_min' : min(G),
        'G_max' : max(G),
        'R_mean' : np.mean(R),
        'R_variance' : np.var(R),
        'R_min' : min(R),
        'R_max' : max(R),
        'RG_mean' : np.mean(R)/np.mean(G),
        'RB_mean' : np.mean(R)/np.mean(B),
        'GB_mean' : np.mean(G)/np.mean(B),
    }
    
    return features
            

def GLCM_LR(img):
    max_img = max(img.flatten())
    GLCM = np.zeros((max_img,max_img))
    
    for x in range(1, img.shape[0]):
        for y in range(0, img.shape[1]):
            GLCM[img[x][y] - 1][img[x-1][y] - 1] += 1           
    return GLCM


# Calculate GLCM matrix using Top-to-bottom method
# @img - source image
#
# @return GLCM - calculated GLCM matrix
def GLCM_TB(img):
    max_img = max(img.flatten())
    GLCM = np.zeros((max_img,max_img))
    
    for x in range(0, img.shape[0]):
        for y in range(1, img.shape[1]):
            GLCM[img[x][y] - 1][img[x][y-1] - 1] += 1
            
    return GLCM


# Calculate GLCM matrix using Top left-to-bottom right method
# @img - source image
#
# @return GLCM - calculated GLCM matrix
def GLCM_TL_BR(img):
    max_img = max(img.flatten())
    GLCM = np.zeros((max_img,max_img))
    
    for x in range(1, img.shape[0]):
        for y in range(1, img.shape[1]):
            GLCM[img[x][y] - 1][img[x-1][y-1] - 1] += 1
                          
    return GLCM


# Calculate GLCM matrix using Top right-to-bottom left method
# @img - source image
#
# @return GLCM - calculated GLCM matrix
def GLCM_TR_BL(img):
    max_img = max(img.flatten())
    GLCM = np.zeros((max_img,max_img))
    
    for x in range(img.shape[0]-2, 0, -1):
        for y in range(img.shape[1]-2, 0, -1):
            GLCM[img[x][y] - 1][img[x+1][y-1] - 1] += 1
                
    return GLCM
    
    
# Calculate homogeneity H
# @glcm - GLCM source matrix
#
# @return H_sum - computed H value
def calc_H(glcm):
    H_sum = 0
    for i in range(0, glcm.shape[0]):
        for j in range(0, glcm.shape[1]):
            H_sum += (glcm[i][j])/(1 + abs(i - j))
    return H_sum


# Calculate correlation Cor
# @glcm - GLCM source matrix
#
# @return Cor_sum - computed Cor value
def calc_Cor(glcm):
    u_i = 0
    u_j = 0
    for i in range(0, glcm.shape[0]):
        for j in range(0, glcm.shape[1]):
            u_i += i*glcm[i][j]
            u_j += j*glcm[i][j]
    
    o_i = 0
    o_j = 0
    for i in range(0, glcm.shape[0]):
        for j in range(0, glcm.shape[1]):
            o_i += ((i - u_i)**2)*glcm[i][j]
            o_j += ((j - u_j)**2)*glcm[i][j]
    
    o_i = math.sqrt(o_i)
    o_j = math.sqrt(o_j)
    
    Cor_sum = 0
    for i in range(0, glcm.shape[0]):
        for j in range(0, glcm.shape[1]):
            Cor_sum += ((i - u_i)*(j - u_j)*glcm[i][j])/(o_i*o_j)
            
    return Cor_sum


# Calculate Contrast Con
# @glcm - GLCM source matrix
#
# @return Con - computed Con value
def calc_Con(glcm):
    Con_sum = 0
    for i in range(0, glcm.shape[0]):
        for j in range(0, glcm.shape[1]):
            Con_sum += (abs(i - j)**2)*glcm[i][j]
    return Con_sum

# Computes differential structures parameters of image
# @img - source image
#
# @return features - dictionary with calculated parameters
def diff_struct_quantification(img):
    
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_glcm_lr = GLCM_LR(img)
    img_glcm_tb = GLCM_TB(img)
    img_glcm_tl_br = GLCM_TL_BR(img)
    img_glcm_tr_bl = GLCM_TR_BL(img)
    
    E_LR = np.sum(img_glcm_lr*img_glcm_lr)
    E_TB = np.sum(img_glcm_tb*img_glcm_tb)
    E_TL_BR = np.sum(img_glcm_tl_br*img_glcm_tl_br)
    E_TR_BL = np.sum(img_glcm_tr_bl*img_glcm_tr_bl)
    H_LR = calc_H(img_glcm_lr)
    H_TB = calc_H(img_glcm_tb)
    H_TL_BR = calc_H(img_glcm_tl_br)
    H_TR_BL = calc_H(img_glcm_tr_bl)
    Cor_LR = calc_Cor(img_glcm_lr)
    Cor_TB = calc_Cor(img_glcm_tb)
    Cor_TL_BR = calc_Cor(img_glcm_tl_br)
    Cor_TR_BL = calc_Cor(img_glcm_tr_bl)
    Con_LR = calc_Con(img_glcm_lr)   
    Con_TB = calc_Con(img_glcm_tb) 
    Con_TL_BR = calc_Con(img_glcm_tl_br)
    Con_TR_BL = calc_Con(img_glcm_tr_bl)
    
    
    features = {
        'E_LR' : E_LR,
        'E_TB' : E_TB,
        'E_TL_BR' : E_TL_BR,
        'E_TR_BL' : E_TR_BL,
        'H_LR' : H_LR,
        'H_TB' : H_TB,
        'H_TL_BR' : H_TL_BR,
        'H_TR_BL' : H_TR_BL,
        'Cor_LR' : Cor_LR,
        'Cor_TB' : Cor_TB,
        'Cor_TL_BR' : Cor_TL_BR,
        'Cor_TR_BL' : Cor_TR_BL,
        'Con_LR' : Con_LR,        
        'Con_TB' : Con_TB, 
        'Con_TL_BR' : Con_TL_BR,
        'Con_TR_BL' : Con_TR_BL,
        'E_mean' : (E_LR + E_TB + E_TL_BR + E_TR_BL)/4,
        'H_mean' : (H_LR + H_TB + H_TL_BR + H_TR_BL)/4,
        'Cor_mean' : (Cor_LR + Cor_TB + Cor_TL_BR + Cor_TR_BL)/4,
        'Con_mean' : (Con_LR + Con_TB + Con_TL_BR + Con_TR_BL)/4,
    }
    
    return features



            
            
            
            
            
            