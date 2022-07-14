# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage as nd
from skimage.filters import sobel
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#%%
img = cv2.imread('E:/Negin Project/entropy.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x_data = np.load('E:/Negin Project/data/X_small.npy')
y_data = np.load('E:/Negin Project/data/y_small.npy')
#%% 
#convert 2D image into single list (array)
df = pd.DataFrame()
count = 0
for x in x_data:
    reshaped = x_data[count].reshape(-1)
    label = 'frame' + str(count)
    df[label] = reshaped
    count+=1
    

#%%
newdf= df.copy()
#print(reshaped.shape)
plt.imshow(x_data[7484], cmap='gray')
plt.show()
#%%
img = df[0]
print(img)

#%%
plt.imshow(x_data[2], cmap='gray')
plt.show()

entropy_img = entropy(reshaped, disk(1))
entropy1 = entropy_img.reshape(-1)
df['Entropy'] = entropy1

entropy_img = entropy(img, disk(1))
entropy1 = entropy_img.reshape(-1)
df['Entropy'] = entropy1

#use entropy filter (lack of order)
cv2.imshow("Original Image", img)




gaussian_img = nd.gaussian_filter(img, sigma=3)
gaussian1 = gaussian_img.reshape(-1)
df['Gaussian'] = gaussian1


sobel_img = sobel(img)
sobel1 = sobel_img.reshape(-1)
df['Sobel'] = sobel1


#gabor filter with orientation of 0 for horizontal features

#cv2.imshow("Entropy", entropy_img)
#cv2.imshow("Gaussian", gaussian_img)
#cv2.imshow("Sobel", sobel_img)
#cv2.waitKey()
#cv2.destroyAllWindows()




