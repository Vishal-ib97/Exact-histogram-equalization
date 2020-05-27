# -*- coding: utf-8 -*-

#The following code was developed in google colab

from PIL import Image
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from urllib.request import urlopen


kernel_0 = np.array([[1]])
kernel_1 = np.array([[0,1,0],[1,1,1],[0,1,0]])
kernel_2 = np.array([[1,1,1],[1,1,1],[1,1,1]])
kernel_3 = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]])
kernel_4 = np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])
kernel_5 = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])

kernel = {5:[kernel_0, kernel_1, kernel_2, kernel_3, kernel_4, kernel_5]}

def sort_matrix(im,k_order):
  q = np.empty([m*n, k_order+1])
  i = 0
  # processing the input image with k averaging kernels
  for ker in kernel[k_order]:
    a = scipy.signal.convolve2d(im,ker, mode = 'same')
    a = a.flatten().T #flatten and transpose
    q[:,i] = a  #appending the average values computed to the columns of q
    i = i+1
  Q = np.empty([m*n, k_order+2])
  for i in range(0, k_order):
    Q[:,i] = q[:,i]
  Q[:,-1] = np.arange(m*n).T #storing the original order in the last column before sorting
  r_Q = np.rot90(Q)
  ind = np.lexsort(r_Q)#Performing a lexicographical sorting 
  sort_Q = Q[ind,:]    # the lexicographically sorted matrix
  return sort_Q

def exactHistEq(im):
  imag = im
  k_order = 5      #assume that the order of pixel intensities no larger than 5x5
  Q = sort_matrix(im,k_order)
  Out = Q[:,-1]
  w = int(0)
  t = int((m*n)/256)
  #assigning values to pixels in the corresponding bins
  for i in range(0,256):
    e = w + t
    Q[w:e,0] = i 
    w = e
  i = 0
  Q = Q[Q[:,-1].argsort()] #reordering in the original pixel order    
  Out = Q[:,-1]
  #reassigning the modified pixel values to the image pixels 
  for h in Out:
    y = int(h%n)
    x = int((h-y)/n)
    imag[x,y] = Q[i,0]
    i += 1
  display_image(imag) #histogram equalized image
  plt.figure()
  plt.hist(imag.flatten(), 256, [0,256])  #equalized histogram
  
if __name__ == "__main__":  
  img_url = '#enter image url'
  img = Image.open(urlopen(img_url))
  img = np.array(img,dtype=int)
  im = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3.0
  display_image(im)  #input image
  plt.figure()
  plt.hist(im.flatten(), 256, [0,256])   #histogram of input image
  m = im.shape[0]
  n = im.shape[1]
  exactHistEq(im)

