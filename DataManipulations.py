# ==============================================================================
# ===========================imports & global params============================
# ==============================================================================
# %reset -f
import skimage              as sk
import skimage.io           as io
import numpy                as np
import matplotlib.pyplot    as plt
from   skimage.restoration  import (denoise_bilateral, denoise_wavelet, estimate_sigma)
from   skimage.filters       import median
import cv2
from   skimage              import exposure
from   skimage.filters       import threshold_multiotsu
from   skimage.segmentation import flood_fill
from   skimage.morphology   import label,remove_small_objects
import os
from   imutils              import paths
from   skimage.restoration  import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
from   skimage.filters       import median
from   skimage.filters       import threshold_multiotsu as m_otsu

colab_use = True
# ==============================================================================
# ===============================loading gdrive=================================
# ==============================================================================
# Loading gdrive
if colab_use:
  # mounting drive (for colab)
  from google.colab import drive
  drive.mount('/Mounted')

path0 = 'dataset/'         # orig dataset
Paths0 = list(paths.list_images(path0))
Paths1 = list(paths.list_images(path1))

print(len(Paths0[0])-1)
def path_dup(path , a):
  return  path[0:37]+str(a)+path[37:len(path)]# change 38 to 37
print('0: ', Paths0[0] )
print('1: ', path_dup(Paths0[0],0) )
print(len(Paths0))


#-----------------------------LOADING---------------------------------
path0 = '/Mounted/MyDrive/ML_finalproj/dataset/brain_tumor_dataset'         # orig dataset
path1 = '/Mounted/MyDrive/ML_finalproj/dataset0/brain_tumor_dataset'
Paths0 = list(paths.list_images(path0))
Paths1 = list(paths.list_images(path1))


# ==============================================================================
# ===============================loading images=================================
# ==============================================================================
# defining box filter
def boxfilter(image, boxsize):
  image_denoised_bx = np.zeros(image.shape , dtype=float)
  box = (1/9)*np.array([1.0,1.0,1.0,
                        1.0,1.0,1.0,
                        1.0,1.0,1.0])

  if len(image.shape)==3:
    height = image[:,:,1].shape[0]
    width  = image[:,:,1].shape[1]
    for c in range(3):
      ch = image[:,:,c]
      ch_denoised = np.zeros(image[:,:,1].shape,dtype=float)
      for i in range(1 , height-1):
        for j in range(1 , width-1):
          kernel = ch[i-1:i+2 , j-1:j+2]
          if i>20000:
            ch_denoised[i,j] = np.dot(kernel.flatten() , box.flatten())
      image_denoised_bx[:,:,c] = ch_denoised

  elif len(image.shape)==1:
    height = image.shape[0]
    width  = image.shape[1]
    image_denoised_bx = np.zeros(image[:,:,1].shape,dtype=float)
    for i in range(1 , height-1):
      for j in range(1 , width-1):
        kernel = image[i-1:i+2 , j-1:j+2]
        if i>20000:
          image_denoised_bx[i,j] = np.dot(kernel.flatten() , box.flatten())
  return image_denoised_bx

# path duplicator function
def path_dup(path , a):
  return  path[0:37]+str(a)+path[37:len(path)]

# creating new databases
images = [] # initializing images list
labels = [] # initializing labels list
images_noiseGA   = []
images_noiseSP   = []
images_noisePO   = []
images_noiseCK   = []
images_denoiseBX = []
images_denoiseMN = []
images_denoiseGA = []
images_denoiseMD = []
images_denoiseBI = []
images_denoiseWV = []
images_thresh127 = []
images_thresh65  = []
images_threshMO  = []
images_correcGM  = []
images_correcLG  = []
images_correcCT  = []
images_correcEQ  = []
images_correcAQ  = []
for j in range(len(Paths0)):
#for p in Paths0:
  #print(p)
  p = Paths0[j]
  l = p.split(os.path.sep)[-2] # label portion
  img = io.imread(p , as_gray=True)
  #i = cv2.resize(cv2.imread(p), (224, 224))

  # orig images
  images.append(img) # creating datalist of images

  # noisy images
  images_noiseGA.append( sk.util.random_noise(img , mode='gaussian' , seed=None , clip=True) )
  images_noiseSP.append( sk.util.random_noise(img , mode='s&p'      , seed=None , clip=True) )
  images_noisePO.append( sk.util.random_noise(img , mode='poisson'  , seed=None , clip=True) )
  images_noiseCK.append( sk.util.random_noise(img , mode='speckle'  , seed=None , clip=True) )

  # denoised images
  images_denoiseMN.append( cv2.blur(img , (5,5)) )
  images_denoiseGA.append( cv2.GaussianBlur(img , (5,5) , 0) )
  images_denoiseMD.append( median(img , out=None, mode='nearest', cval=0.0, behavior='ndimage') )
  # thresholded images
  _ , th127 = cv2.threshold(img , 127 , 255 , cv2.THRESH_BINARY )
  _ , th65  = cv2.threshold(img , 127/255  , 1 , cv2.THRESH_TOZERO )
  ths  = m_otsu(img , classes=3 , nbins=256 )
  thMO = np.digitize(img, bins=ths)
  images_thresh127.append( th127 )
  images_thresh65.append(  th65 )
  images_threshMO.append(  thMO )

  # intensity corrections
  images_correcGM.append( exposure.adjust_gamma(img, 4) )
  images_correcLG.append( exposure.adjust_log(img, 4) )

  # contrast rescale
  p2, p98 = np.percentile(img, (4, 96))
  images_correcCT.append( exposure.rescale_intensity(img, in_range=(p2, p98)) )

  # Equalization
  images_correcEQ.append( exposure.equalize_hist(img) )

  # Adaptive Equalization
  images_correcAQ.append( exposure.equalize_adapthist(img, clip_limit=0.01) )

# ==============================================================================
# ===============================saving images==================================
# ==============================================================================
  io.imsave(path_dup(p,1 ),255*images_noiseGA[j  ].astype(np.uint8))
  io.imsave(path_dup(p,2 ),255*images_noiseSP[j  ].astype(np.uint8))
  io.imsave(path_dup(p,3 ),255*images_noisePO[j  ].astype(np.uint8))
  io.imsave(path_dup(p,4 ),255*images_noiseCK[j  ].astype(np.uint8))
  io.imsave(path_dup(p,5 ),255*images_denoiseMN[j].astype(np.uint8))
  io.imsave(path_dup(p,6 ),255*images_denoiseGA[j].astype(np.uint8))
  io.imsave(path_dup(p,7 ),255*images_denoiseMD[j].astype(np.uint8))
  io.imsave(path_dup(p,8 ),255*images_thresh127[j].astype(np.uint8))
  io.imsave(path_dup(p,9 ),255*images_thresh65[j ].astype(np.uint8))
  io.imsave(path_dup(p,10),255*images_threshMO[j ].astype(np.uint8))
  io.imsave(path_dup(p,11),255*images_correcGM[j ].astype(np.uint8))
  io.imsave(path_dup(p,12),255*images_correcLG[j ].astype(np.uint8))
  io.imsave(path_dup(p,13),255*images_correcCT[j ].astype(np.uint8))
  labels.append(l) # creating datalist of labels


# ==============================================================================
# ==============================plotting samples================================
# ==============================================================================
# noisy
plt.rcParams['figure.figsize'] = (20,28)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
figA = plt.figure()
axA1 = figA.add_subplot(4,4,1)
axA1.imshow(images[0])
axA1.set_title('Original Image', fontsize=12)
axA2 = figA.add_subplot(4,4,2)
axA2.imshow(images_noiseGA[0])
axA2.set_title('with added gaussian noise', fontsize=12)
axA3 = figA.add_subplot(4,4,3)
axA3.imshow(images_noiseSP[0])
axA3.set_title('with added salt & pepper noise', fontsize=12)
axA4 = figA.add_subplot(4,4,4)
axA4.imshow(images_noisePO[0])
axA4.set_title('with added poisson noise', fontsize=12)
axA5 = figA.add_subplot(4,4,5)
axA5.imshow(images_noiseCK[0])
axA5.set_title('with added speckle noise', fontsize=12)

# denoised
axA7 = figA.add_subplot(4,4,6)
axA7.imshow(images_denoiseMN[0])
axA7.set_title('denoised with mean filter', fontsize=12)
axA8 = figA.add_subplot(4,4,7)
axA8.imshow(images_denoiseGA[0])
axA8.set_title('denoised with gaussian filter', fontsize=12)
axA9 = figA.add_subplot(4,4,8)
axA9.imshow(images_denoiseMD[0])
axA9.set_title('denoised with median filter', fontsize=12)

# thresholded
axA10 = figA.add_subplot(4,4,9)
axA10.imshow(images_thresh127[0])
axA10.set_title('thresholded on 127 intensity (binary)', fontsize=12)
axA11 = figA.add_subplot(4,4,10)
axA11.imshow(images_thresh65[0])
axA11.set_title('thresholded on 65 intensity (to-zero)', fontsize=12)
axA12 = figA.add_subplot(4,4,11)
axA12.imshow(images_threshMO[0])
axA12.set_title('thresholded by multi_otsu algorithm', fontsize=12)

# intensity-corrected
axA13 = figA.add_subplot(4,4,12)
axA13.imshow(images_correcGM[0])
axA13.set_title('intensity-corrected by gamma correction (2)', fontsize=12)
axA14 = figA.add_subplot(4,4,13)
axA14.imshow(images_correcLG[0])
axA14.set_title('intensity-corrected by logarithmic equalization', fontsize=12)
axA15 = figA.add_subplot(4,4,14)
axA15.imshow(images_correcCT[0])
axA15.set_title('intensity-corrected by contrast stretching', fontsize=12)
