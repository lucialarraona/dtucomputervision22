#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:34:30 2022

@author: lucialarraona
"""


################################3 The eight point algorihm ##################33
import numpy as np 

homo3d_to_inhomo2d = lambda x: x[:-1]/x[-1]
inhomo2d_to_homo3d = lambda x: np.r_[x,np.transpose(np.ones((x.shape[1],1)))]

#%%  9.1 Estimates a fundamental matrix from eight or more point correspondences, using the linear algorithm.

## Fundamental matrix from week3, call it Ftrue 

FTrue = F

# Load the points
qs = np.load('/Users/lucialarraona/Desktop/DTU/Computer Vision/Week9_CV/qs.npy', allow_pickle =True)

q1 =  qs.item(0)['q1']
q2 = qs.item(0)['q2']




def Fest_8point(q1, q2):
    
    size = q1.shape[1] if q1.shape[1] == q2.shape[1]  else print("length of q1 and q2 not equal!")
    #Construct B
    Bi = np.vstack([[q1[0][i]*q2[0][i], q1[0][i]*q2[1][i], q1[0][i], q1[1][i]*q2[0][i], q1[1][i]*q2[1][i], q1[1][i], q2[0][i], q2[1][i], 1] for i in range(size)])
    print(Bi.shape)
    # Solve equation to find F 
    u,s,vh = np.linalg.svd(Bi)
    F = vh[-1,:]
    F = vh[-1,:].reshape((3,3), order = 'F') ## don't forget to reshape the matrix 
    
    # Solve equation 
    
    return F


F_est_sol = Fest_8point(q1,q2)


### WE HAVE TO SCALATE IT TO MAKE IT SIMILAR TO WEEK 3.8 SOLUTION
scalefactor = F[0][0]/ F_est_sol[0][0]

F_est_sol = F_est_sol*scalefactor

#%% 9.2 Repeat part of the exercise from last week, by matching the two images from TwoImageData.npy

import cv2 as cv2 
import matplotlib.pyplot as plt

data = np.load('/Users/lucialarraona/Desktop/DTU/Computer Vision/Excercises/Week3_CV_copia/TwoImageData.npy',allow_pickle = True).item()

img1 = data['im1']
img2 = data['im2']

#img1 = cv2.imread(img1,cv2.IMREAD_GRAYSCALE)  # queryImage
#img2 = cv2.imread(img2,cv2.IMREAD_GRAYSCALE) # trainImage
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

des1 = np.uint8(des1)
des2 = np.uint8(des2)
# BFMatcher with default params
bf = cv2.BFMatcher_create(crossCheck=True)

matches = bf.match(des1,des2)

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:3],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.rcParams["figure.figsize"] = (10,10)
plt.imshow(img3),plt.show()



#%% 9.3 


#### Imports from week7

def setofInliers(line,set_points,delta):
    
    counter = 0 
    set_inliers = []
    
    for p in set_points.T:
        
        if isOutlier(line, p, delta) == False:
            counter += 1 
            set_inliers.append(p)
        
            
        else:
            counter = counter
    
    return counter, np.stack(set_inliers).T




#line_best_final, set_in_final = RANSAC(points,delta =3,N=10)

def pca_line(x): #assumes x is a (2 x n) array of points

    d = np.cov(x)[:, 0] 
    d /= np.linalg.norm(d) 
    l = [d[1], -d[0]] 
    l.append(-(l@x.mean(1))) 
    return l


#line_fit_inliers = pca_line(set_in_final)


#################################### RANSAC ALGORITHM from WEEK7 ######################################

def RANSAC(set_points, delta, N):
    
    
    max_val_count = 0
    
    for _ in range(N):
        
        # Select 2 random points from the set
        p1, p2 = randPoints(set_points)
        
        # Estimate the first line 
        line = estimateLineHomo(p1, p2)
        
        # Count inliers 
      
        #count = countInliers(line, set_points, delta)
        count,set_in = setofInliers(line, set_points, delta)
        
        # If the number of inliers is bigger than what I previously had as max, that is the best value yet
        if count > max_val_count:
            p1_sol = p1
            p2_sol= p2
            set_in_final = set_in
            max_val_count = count # update max value
            line_best = line # save best_line 
   
    print("Max number of inliers=", max_val_count)
    
    plt.scatter(set_points[0,:],set_points[1,:])
    
    x=np.array([p1_sol[0],p2_sol[0]])
    y=np.array([p1_sol[1],p2_sol[1]])
    plt.plot(x,y, color = 'r')
        
    return line_best, set_in

#line_best, set_in = RANSAC(points, delta=3, N=3)


#%%  Modification for the RANSAC algorithm 

# define random matches

matches = np.random.choice(matches, 8, replace=False)

#get the coordinates from the matches? 

list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
list_kp2 = [kp2[mat.queryIdx].pt for mat in matches]


q11 = np.array(list_kp1).T
q22 = np.array(list_kp2).T


# apply those matches to the function

F = Fest_8point(q11, q22) 

# define function to calculate sampson's distance
    
def SampsonsDistance(F, p1, p2):
    
    p1h=inhomo2d_to_homo3d(p1)
    p2h=inhomo2d_to_homo3d(p2)
    
    
    size = p1h.shape[1] if p1h.shape[1] == p2h.shape[1]  else print("length of q1 and q2 not equal!")
    
    D = np.vstack([[(p2h[:,i].T@F@p1h[:,i])**2/
                    ((p2h[:,i].T@F)[0]**2+
                     ((p2h[:,i].T@F)[1]**2)+
                     ((F@p1h[:,i])[0]**2)+
                     ((F@p1h[:,i])[1]**2))
                     ] for i in range(size)])
    
    return D


D = SampsonsDistance(F, q11, q22)
print(D)


threshold = 3.84*(3**2)





#%% New Ransac 

def RANSAC(im1,im2, delta, N):
    
 ## define random matches between pictures 
 
 
 
    