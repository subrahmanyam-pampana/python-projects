# IMAGE PROCESSING USING PYTHON OPEN CV LIBRARY
## INSTALLATION OF NUMPY AND OPEN CV
1. First install `pipwin` if it not installed already. it is the offcial installer for installing `numpy` or any other packeages from unofficial location [unoffcila py libs](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
2. For installing pipwin use `pip install pipwin`
3. After installing pipwin install numpy library using `pipwin install numpy`. it Installs the latest version of numpy compatble [numpy lib](https://download.lfd.uci.edu/pythonlibs/x6hvwk7i/numpy-1.22.2+mkl-cp310-cp310-win_amd64.whl) with currect version of python.
4. Install Open cv library by running `pipwin install opencv-python`. or download it from [open cv](https://download.lfd.uci.edu/pythonlibs/x6hvwk7i/opencv_python-4.5.5+mkl-cp310-cp310-win_amd64.whl)
5. Library name `opencv_python-4.5.5+mkl-cp310-cp310-win_amd64.whl`

6. For testing the installed packages, run the below commands in cmd promt one by one. if packages correctly installed, it will return version
  ```
  import numpy
  numpy.__version__
  
  import cv2
  cv2.__version__
  ```
## 1.BASICS OF OPEN CV(Computer vision)
### 1.1 Reading and Writing the Image
```
import numpy as np
import cv2
#read the image
#below method takes two parameters. 1. image path, 2. color desity, 1 means full original color, 0 means black and white
img = cv2.imread("myImagePath.png",1)

#namaed window for showing the image
#belowe method takes 2 params, 1. name of the window 2. behaviour of the window. 
#we can use 0 for default behaviour or cv2.WINDOW_NORMAL
cv2.namedWindow("my Image Window",cv2.WINDOW_NORMAL)

#Show the image 
#below method takes 2 params. 1. name of the window 2.image byte code 
cv2.imshow("my Image Window",img)

#wait untill the user interact with the interface to close the window
#below method takes one param. 1. number of milliseconds to wait before closing the window.
#if we pass Zero, it will wait untill the user press any Key and close window
cv2.waitKey(0)

#writing the image to current folder
#below method takes two params. 1. new image name.<valid image format> 2. image byte code
#Note: below code convert the input png image to jpeg image. size of the both images different
cv2.imwrite("newImage.jpg",img)
```
### 1.2 Access and Understand Pixel Data
- Image Data is stored as N Dimension Array Contains Rows, Columns 
- Each pixel is ranked with `BGR` color range arrays. Ex: [255,2555,255]
- Pixels follows BGR format
- Important Parameters
    *  `img.shape` gives `(length_Rows,length_columns,num_channels)`
    *  `img.dtype` gives `dtype('uint8')`.unsigned integer of value 8. Which means there are maximum of 2 power 8 values in each pixel. i.e 0 - 255
    *  `img.size` gives `total number of pixels`
- We can also use list slising to Access the Image Pixels
  * For accessing the First channel use `myImage[:,:,0]`
  * For accessing any pixes use `myImage[Row,col]` Ex: `myImage[12,15]`
### 1.3 Data Types and Structes
```
#creating different Images with numpy arrays
import numpy as np
import cv2
#below zeros method of numpy create an array of width 150, height 200 and 1 channel
#zeros method takes two parameter 1. [width,height,num_channels] 2. dtype
black = np.zeros([150,200,1],'uint8')
cv2.imshow("Black",black)

#Show ones image, it almost like black image bcz its maximum value is 255, but we filled the array with all 1's
ones = np.ones([150,200,3],'uint8')
cv2.imshow("One",ones)

#Showing white image
white  = np.ones([150,200,3],'uint16')
#assigning all the values in an array white with maximum value of 'uint16' i.e 65535
white *= (2**16-1)
cv2.imshow("White",white)

#Show Blue color Image
color = ones.copy()
color[:,:] = (255,0,0)
cv2.imshow("Blue",color)

cv2.waitKey(0)

#destroy All the windows
cv2.destroyAllWindows()
```
### 1.4 Image types and Colors
```
import numpy as np
import cv2

img = cv2.imread("butterfly.jpg",1)

cv2.imshow("Image_Tab",img)
cv2.moveWindow("Image_Tab",0,0)

height, width, channels = img.shape
b,g,r  = cv2.split(img)

#concatenate method concatenate the provided arrays based on axis. 
# if axis = 1, then concatenate along width
# if axis = 0, then concatenate along height
rgb_split = np.concatenate((r,g,b),axis=1)
# rgb_split = np.empty([height,width*3,channels],'uint8')
# rgb_split[:,0:width] = cv2.merge([b,b,b])
# rgb_split[:,width:width*2] = cv2.merge([g,g,g])
# rgb_split[:,width*2:width*3] = cv2.merge([r,r,r])

cv2.imshow("channles",rgb_split)
cv2.moveWindow("channles",0,height)

#hsv space
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
hsv_split = np.concatenate((h,s,v),axis=1)
cv2.imshow("hsv split channels", hsv_split)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 1.5 Pixel manupulation and Filtering 
Note:
- jpg images does not supports the alpha channel.Event when we export image as jpg having 4 channels, it again fall back to 3 channels
-  only png images support it.
```
import numpy as np
import cv2

img = cv2.imread("butterfly.jpg",1)

gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imwrite("gray.jpg",gray)
# extract the each channel, we can also use b,g,r = cv2.split(img);
# but below is the fast and efficent way of extracting the channels
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]

rgba = cv2.merge((b,g,r,g)) #here aplha value passed as green, which means non green parts of the image are become transparent
cv2.imwrite("rgba.png",rgba)
```
### 1.6 Gaussian Blur, Dilation and Erosion
  -  These operations help reduce noise or unwanted variances of an image or threshold. The goal is to make the image easier to work with. 
#### The Gaussian Blur
this filter smooths an image by averaging pixel values with its neighbors. It's called a Gaussian Blur because the average has a Gaussian falloff effect. In other words, pixels that are closer to the target pixel have a higher impact with the average than pixels that are far away. This is how the smoothing works. It is often used as a decent way to smooth out noise in an image as a precursor to other processing. 
-  In this method, instead of a box filter, a Gaussian kernel is used. It is done with the function, `cv2.GaussianBlur()`. We should specify the width and height of the kernel which should be **positive and odd**. We also should specify the standard deviation in the X and Y directions, sigmaX and sigmaY respectively. If only sigmaX is specified, sigmaY is taken as the same as sigmaX. If both are given as zeros, they are calculated from the kernel size. Gaussian blurring is highly effective in removing Gaussian noise from an image.
     - `blur = cv.GaussianBlur(img,(sigmaX,sigmaY),0)`
 ```
import numpy as np
import cv2

img = cv2.imread("butterfly.jpg",1)

cv2.imshow("Original_image",img)
cv2.moveWindow("Original_image",1,1)
height,width,channels = img.shape

blurImg = cv2.GaussianBlur(img,(5,55),0)

cv2.imshow("Blur_Image",blurImg)
cv2.moveWindow("Blur_Image",1,height)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
#### Dilation and Erosion:
Dilation adds pixels to the boundaries of objects in an image, while erosion removes pixels on object boundaries. The number of pixels added or removed from the objects in an image depends on the size and shape of the structuring element used to process the image.
- `Dilation:` The value of the output pixel is the `maximum value` of all pixels in the neighborhood. In a binary image, a pixel is set to 1 if any of the neighboring pixels have the value 1. Morphological dilation makes objects more visible and fills in small holes in objects.
- `Erosion:` The value of the output pixel is the `minimum value` of all pixels in the neighborhood. In a binary image, a pixel is set to 0 if any of the neighboring pixels have the value 0. Morphological erosion removes islands and small objects so that only substantive objects remain. [to know more click here](https://www.mathworks.com/help/images/morphological-dilation-and-erosion.html#:~:text=Dilation%20adds%20pixels%20to%20the,used%20to%20process%20the%20image) 
```
import numpy as np
import cv2

img = cv2.imread("map.jpg",1)

cv2.imshow("Original_image",img)
cv2.moveWindow("Original_image",1,1)
height,width,channels = img.shape

kernel = np.ones((5,5),'uint8')

dilateImg = cv2.dilate(img,kernel,iterations=1)
erodeImg = cv2.erode(img,kernel,iterations=1)

cv2.imshow("Dilate Image",dilateImg)
cv2.imshow("Erode Image",erodeImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
 ### 1.7 Scale and Rotate Image
 #### Scale Image code
 ```
 from configparser import Interpolation
import numpy as np
import cv2

img = cv2.imread("butterfly.jpg",1)

cv2.imshow("Original_image",img)
height,width,channels = img.shape

#scaling the Image
#half image without disturbing the original image
half_img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
stretch_img = cv2.resize(img,(600,600))
stretch_img_near = cv2.resize(img,(600,600),interpolation=cv2.INTER_NEAREST)
cv2.imshow("half image",half_img)
cv2.imshow("stretch_img",stretch_img)
cv2.imshow("stretch_img_near",stretch_img_near)

cv2.waitKey(0)
cv2.destroyAllWindows()
 ```
 #### Rotate Image Code
 - Capital letters are user to represent the Matrix
 - `cv2.warpAffine(actual_image, rotationMatrix, (width,height))` method applies Matrix transformation on actual image
 ```
import numpy as np
import cv2

img = cv2.imread("butterfly.jpg",1)

cv2.imshow("Original_image",img)
height,width,channels = img.shape

M = cv2.getRotationMatrix2D((width//2,height//2),-90,1) #(possition,degrees,1)
rotated_img = cv2.warpAffine(img,M,(width,height)) #(actual_image, rotationMatrix, (width,height))

cv2.imshow("rotatedImage",rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 ```
 ### 1.8 Video Inputs
`cv2.VideoCapture(0)` method used to capture the Video Input.
 ```
import numpy as np
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    cv2.imshow("Frame",frame)
    ch = cv2.waitKey(1) #wait 1 milli sec
    if ch & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
 ```
### 1.9 Draw Circle On the Video at Mouse Click On the video
- `cv2.circle(frame,point,radius,color,line_width)` method Draws the circle on the Defined window
- `cv2.setMouseCallback("Frame",click)` method to register the click event to Frame. here click is the call back function
```
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

#circle Parameters
radius = 50
color = (0,255,0)
line_width = 1
point = (100,100)

def click(event,x,y,flags,param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",click)

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    cv2.circle(frame,point,radius,color,line_width)
    cv2.imshow("Frame",frame)
    ch = cv2.waitKey(1) #wait 1 milli sec
    if ch & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
### 1.1 White Board App
- press `b` to change color to Blue
- press `g` to change color to green
- press `c` to clear the screen
- left click and hold to write
```
import numpy as np
import cv2
from pyparsing import col

canvas = np.ones((500,500,3),'uint8')*255
pressed = False

#circle Parameters
radius = 3
color = (0,255,0) #green color
line_width = -1 # -1 for filling the circle
def click(event,x,y,flags,param):
    global point,color,canvas,pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        pressed = True
    elif event == cv2.EVENT_MOUSEMOVE and pressed==True:
        point = (x,y)
        cv2.circle(canvas,point,radius,color,line_width) 
    elif event == cv2.EVENT_LBUTTONUP:
        pressed = False


cv2.namedWindow("Canvas")
cv2.setMouseCallback("Canvas",click)

while True:
    cv2.imshow("Canvas",canvas)
    ch = cv2.waitKey(1) #wait 1 milli sec
    if ch & 0xFF == ord('q'):
        break
    elif ch & 0xFF == ord('b'):
        color = (255,0,0)
    elif ch & 0xFF == ord('g'):  
         color = (0,255,0)
    elif  ch & 0xFF == ord('c'):   
        canvas = np.ones((500,500,3),'uint8')*255 
cv2.destroyAllWindows()
```
## 2.Object Detection
### 2.1 Segmentation and Binary Images
-  In this chapter, we are focused on extracting features and objects from images.
-  An `object` is the focus of our processing. It's the thing that we actually want to get, to do further work.
-  In order to get the object out of an image, we need to go through a process called `segmentation`. 
-  `Segmentation` can be done through a variety of different ways but the typical output is a `binary image`. 
-  A `binary image` is something that has values of zero or one. Essentially, a one indicates the piece of the image that we want to use and a zero is everything else. 
-  Binary images are a key component of many image processing algorithms. These are pure, non alias black and white images, the results of extracting out only what you need. They act as a mask for the area of the sourced image. 
-  After creating a binary image from the source, you do a lot when it comes to image processing. 
-  One of the typical ways to get a binary image, is to use what's called the `thresholding algorithm`. This is a type of segmentation that does a look at the values of the sourced image and a comparison against one's central value to decide whether a single pixel or group of pixels should have values zero or one. 
   ![image](https://user-images.githubusercontent.com/79074273/155265947-55f3b7cf-2c0d-4984-921d-550ebb7b4884.png)
-  In the top example on this slide, we can see that the binary threshold applied, looks at every pixel in the image on the left, to see if the value is greater than or equal to **128**. If it is, it is assigned the value of one. Anything less than that is assigned the value of zero. 
-  On the bottom example, the binary threshold is based around the value of 64. Therefore, everything that has a value of **64** on the left hand image, is assigned a one. This is a very simple example of segmentation.

### 2.2 Simple Treshoulding code
```
import numpy as np
import cv2

#black and white image
bwImg = cv2.imread("butterfly.jpg",0)

cv2.imshow("Original Image",bwImg)

height,width = bwImg.shape[0:2]

binaryImg = np.zeros([height,width,1],'uint8')
threshold = 85
#Treshold method, Bruteforce approach
for row in range(0,height):
    for col in range(0,width):
        if bwImg[row][col]>= threshold:
            binaryImg[row][col] = 255            
cv2.imshow("Treshold image", binaryImg)

#open cv optimized method for treshoulding
rect, threshImg = cv2.threshold(bwImg,threshold,255,cv2.THRESH_BINARY)
cv2.imshow("Open cv Treshold image", threshImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 2.4 Adaptive Thresholding
- Simple Thresholding won't work correctly in uneven lightining condition
- unlinke Simple threshold which takes fixed threshold value, adaptive thresholding is applied based on calculated neighborhood threshold value
- `adapt_tresh = cv2.adaptiveThreshold(bwImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)` 
```
from sqlite3 import adapt
import numpy as np
import cv2

#black and white image
bwImg = cv2.imread("butterfly.jpg",0)
cv2.imshow("Original Image",bwImg)

threshold = 85

#open cv optimized method for treshoulding
rect, threshImg = cv2.threshold(bwImg,threshold,255,cv2.THRESH_BINARY)
cv2.imshow("Treshold Basic", threshImg)

#Adaptive Threshold method
adapt_tresh = cv2.adaptiveThreshold(bwImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
cv2.imshow("Adaptive Treshold", adapt_tresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 2.5 Skin Detection
- Skin detection can be done by doing `bitwise_and` Operation betwen min_saturation matrix and max_inverted_hue matrix
- `cv2.bitwise_and(min_sat,max_hue)`
```
import numpy as np
import cv2

img = cv2.imread("faces.jpg",1)
img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
cv2.imshow("Original Image",img)

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h,s,v = hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]

hsv_split = np.concatenate((h,s,v),axis=1)
# cv2.imshow("hsv split Image",hsv_split)

ret,min_sat = cv2.threshold(s,40,255,cv2.THRESH_BINARY)

ret,max_hue = cv2.threshold(h,35,255,cv2.THRESH_BINARY_INV) #THRESH_BINARY_INV inverts the black and white pixels

conv_img = cv2.bitwise_and(min_sat,max_hue) #convoluted image

finalImg = np.concatenate((min_sat,max_hue,conv_img),axis=1)
cv2.imshow("final image",finalImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 2.6 Contours
#### 2.6.1 Introduction to Contours
- Once we have segmented out the key areas of an image, the next step is typically to identify the individual objects. But how can we do that? 
- One powerful way is to use OpenCV's implementation of contours. The goal of contours is to take a binary image and create a tightly fitting closed perimeter around all individual objects in the scene. Each perimeter is called a `contour`. 
- From a mathematical point of view, it is called an `iterative energy reduction algorithm`. But conceptually, we can think of it as an elastic film that starts on the edges of an image and squeezes in around all the objects and shapes. It creates the boundary around all these objects. 
![Untitled](https://user-images.githubusercontent.com/79074273/155276604-a4ab9a2f-c73f-4c06-a8f6-2ce64d8154ca.png)
- One thing to be aware of is the idea of neighborhoods and connectedness. Contours will consider any pixel value above zero as part of the foreground, and any other pixels touching or connected to this pixel will be made to be part of the same object. As the algorithm runs, it tries to reduce the energy or the bounding box around all these objects until it comes to a converged result. 
- It's important to understand that while this may be an iterative algorithm, we know contours will always converge, so it'll never be stuck in an infinite loop. At the end, you have a list of contours, and each contour is simply a linear list of points which describe the perimeter of a single object. They are always enclosed, a technical term meaning there are no gaps. This means they can be safely drawn back onto an image and completely filled with a new color. 
- Contours is one of the gateways to determine many other useful properties about a single object within an image, making it a very powerful algorithm at our image processing disposal. It moves from the step of object segmentation, often done by thresholding, into the step of object detection.

#### 2.6.2 Contour Object Detection 
- `contours,hierarchy = cv2.findContours(thresholdImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)` Method gives list of available objects contours and thier heirarchy
- `cv2.drawContours(original_image,contours_list,index,color,thickness)` method used to draw the contors on the image. for drawing all the contours pass index as -1.
```
import numpy as np
import cv2

img = cv2.imread("objects.png",1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

tresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)

contours,hierarchy = cv2.findContours(tresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img2  = img.copy()
index,color,thickness = -1,(255,0,255),4

cv2.drawContours(img2,contours,index,color,thickness)

cv2.imshow("Original Image",img)
cv2.imshow("contour Image",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**Result**
![image](https://user-images.githubusercontent.com/79074273/155279662-41b93460-f5a6-4824-b731-713b378f8d17.png)

#### 2.6.3 Finding Area, perimeter, center, and curvature of Contours
- `area = cv2.contourArea(single_contour)` method return the area of the contour
- `perimeter = cv2.arcLength(single_contour,closed_open?)` method return the perimenter of the contour. here `closed_open` is `boolean` value. `True` indicates the `Closed contour`
- center can be found by using moments
```
M = cv2.moments(single_contour)
cx = int( M['m10']/M['m00'])
cy = int( M['m01']/M['m00'])
```
- `cv2.circle(img, (cx,cy),radius,color,thickness)` method used to draw the circle. 
```
import numpy as np
import cv2

img = cv2.imread("objects.png",1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

tresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)

contours,hierarchy = cv2.findContours(tresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img2  = img.copy()
index,color,thickness = -1,[111,111,111],4

objects = np.zeros([img.shape[0],img.shape[1],3],'uint8')

for i in range(1,len(contours)):
  c = contours[i]
  color = [x+34 for x in color]
  cv2.drawContours(objects,[c],index,color,-1)

  perimeter = cv2.arcLength(c,True)
  area = cv2.contourArea(c)

  M = cv2.moments(c)
  if not (M['m10']<1 or M['m01']<1 or M['m00']<1):
    cx,cy = int( M['m10']//M['m00']), int(M['m01']//M['m00']) 
    cv2.circle(objects,(cx,cy),4,(0,0,255),-1)

cv2.imshow("Original Image",img)
cv2.imshow("contour Image",objects)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 2.7 Canny Edge detection Algorithm
- Often we need to pre-process images in order to improve our final result, and in the case of extracting contours from individual objects in an image it is often handy to first detect and accentuate the edges within an image. 
- Canny Edges is one type of edge detection algorithm that works quite well to help create better separation of objects within the image. Generally speaking, edge detection algorithms look at the rate or speed at which color changes across the image. 
- Canny Edges is a specific form of that algorithm that creates a single pixel wide line at key high gradient areas in the image. This can help break up the object if there was an overlap in our segmentation process. 
- Let's take a look at how we can use Canny Edges. Imagine the goal here is to try and segment out each individual tomato. If we're running a threshold, we may run into an issue where the different tomatoes get blobbed together as one single object. 
- If we put these two images side by side, we can see we have a lot more detail about the edges in the Canny image. You can imagine one way that we could improve our segmentation would be to take a `difference between the Canny and the threshold`. 
- If you take a look at the tomato on the left hand threshold image, you can see it's joined together with the other tomatoes, but the lines in the Canny image, if you were to make that take away form the threshold image, it would actually break those up into separate objects. This is just one of many use cases of Canny image and in general edge detection algorithms.
- `canny = cv2.Canny(img,max_thresh,min_thresh)`

![image](https://user-images.githubusercontent.com/79074273/155450449-7c0c94d1-c729-4b8c-a938-b97f085f1881.png)

```
import numpy as np
import cv2

img = cv2.imread("tamatoes.jpg",1)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow("original image",img)
ret,thresh = cv2.threshold(hsv[:,:,0],25,255,cv2.THRESH_BINARY_INV)
cv2.imshow("Thresh image",thresh)

canny = cv2.Canny(img,100,70)
cv2.imshow("canny image",canny)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 2.8 Object Detection Overview
- In this chapter, we reviewed a few ways to approach `segmenting out objects in an image` and `detecting properties of those objects`. 
- A few areas we looked at included both `simple` and `adaptive thresholding` using edges to help break down apart closely fitting objects. 
- We also briefly looked at how to composite multiple thresholds of different types together, and in the last chapter, we saw how to use `Gaussian blurs` to reduce noise, and `dilation` and `erosion` filters to reduce small speckles or gaps. 
- These are just some of the image processing tools helpful in segmenting out objects. It's important to keep in mind the context. Know what the application will be used for, and develop segmentation that will fit the use case. 
- Do you know that your lighting will always stay `roughly the same` for different image inputs? If so, it may be more effective to use `non-adaptive thresholding`; 
- perhaps you can improve your thresholding by gathering your own global average or mean. 
- How about object orientation and scale? Can you make assumptions about the size of an object in an image, therefore allowing you to filter out anything that doesn't fit that size? Furthermore, is it a real-time application, where consistency between frames might be very important? As for filtering, is it a problem to over filter or under filter the results? For example, if detecting an object triggers an action, such as sending email, then perhaps you want to be more sensitive about not having false positives. 
- Though this course only covers a few basic ways to identify and characterize image objects, there are many other applications and advanced techniques to segment out objects in an image or sequence. In the last slide, we touched upon a priori means of object detection. This means having some prior knowledge about the context or inputs ahead of time. For example, if you know that the subjects of your image will always be against a black background, it allows you to make different assumptions and processing technique decisions that wouldn't hold up in more general situations. When looking to detect and segment out objects in an image, be aware of all the tools at your disposal. Know how the parameters of the situation will vary, and think about how you can break the process down into smaller steps.

![image](https://user-images.githubusercontent.com/79074273/155450834-ab6d7e20-9e22-4e17-bbb3-7f2a52940de6.png)

![image](https://user-images.githubusercontent.com/79074273/155452271-cd90a255-55a6-4b47-85ea-00636886b841.png)

### 2.9 Project: Draw all the contour whose area graeter than three thousand of fuzzy image
- first convert the fuzzy image to gray image
- blur the noise using `gussian blur`. which heighlights the target area and blurs the uncesseary portion
- if we apply the `simple threshold filter` on the blur image, it becomes almost white image, so apply the `inverse adaptive threshod filter`.
- if we apply the `direct adaptive filter`, targetted area becomes **black** and unwanter area becomes **white**. but in order to get the contour of targetted area invert black and white.
- thats why here applied `inverse adaptive threshod filter` for finding the `contours`
- filtered the contours which is having area less than three thousand

```
import numpy as np
import cv2
import random

img = cv2.imread("fuzzy.png",1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Original Image",img)

blur = cv2.GaussianBlur(gray,(3,3),0)
cv2.imshow("blur Image",blur)

ret, simple_thresh = cv2.threshold(blur,45,255,cv2.THRESH_BINARY)
cv2.imshow("Simple thresh Image",simple_thresh)

tresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
cv2.imshow("adaptive thresh Image",tresh)

tresh_inv = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,115,1)
cv2.imshow("adaptive thresh inv Image",tresh_inv)

contours,hierarchy = cv2.findContours(tresh_inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours2,hierarchy2 = cv2.findContours(tresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img2  = np.ones([img.shape[0],img.shape[1],3],'uint8')*255
img3  = np.ones([img.shape[0],img.shape[1],3],'uint8')*255

index,thickness = -1,-1

for i in range(1,len(contours)):
  c = contours[i]
  area = cv2.contourArea(c)
  color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
  if(area>3000):
    print("area:",area)
    cv2.drawContours(img2,[contours[i]],index,color,thickness)

for i in range(1,len(contours2)):
  c = contours2[i]
  area = cv2.contourArea(c)
  color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
  if(area>3000):
    print("area:",area)
    cv2.drawContours(img2,[c],index,color,thickness)

cv2.imshow("contour using inv adapive tresh Image",img2)
cv2.imshow("contour using adapive tresh Image",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
Result: 
![image](https://user-images.githubusercontent.com/79074273/155465517-989a79c7-c37a-47f8-9d35-02e32d785c95.png)

## 3. Face and Feature Detection
### 3.1 Introduction to Face and Feature Detection
- This chapter is dedicated to feature recognition and face detection. 
- A feature is any part of an image which can be used to understand an image better and to help perform a needed task. 
- Features can be visual components of an image or more mathematical properties of the patterns and arrangement of the pixel values. 
- Good features are invariant to changes like lighting or scaling. For example, the circularity of a segmented object or the ratio of the shortest to the longest axis.
- Statistical properties of an image or color distribution could also act as features if it fits the problem. Features are identified by classifiers to either detect or recognize objects in an image. 
- Note to be careful how you use the term detection versus recognition. 
- `Detection` is often the step prior to recognition. For example, with faces, you first might detect whether a face exists in an image, and a follow-up step might be to understand if that face matches any image in a database. In this case, `features` such as the distance between eyes in an image may be used both for the detection process to see whether or not a face actually exists, but also used as a classifier for the `recognition process` to see which face it specifically matches with. 
- In this chapter, we will be looking at two specific algorithms, `template matching` for general object recognition, and `hard cascading` as a means for face detection.

### 3.2 Introduction to template matching
- When it comes to feature detection, template matching is a readily available and straightforward method. The way template matching works, is it searches for a similar pattern between two images. 
- This is accomplished by taking a reference image, called a `template` and sliding it around the other `comparison image`, taking a difference at every position. The result, is a black and white gray scale image with varying intensities showing how well it matched at each position. 
- Using a 1D, somewhat modified example, we can see that as we slide our template, which is a triangle, across the screen, it does a comparison in each spot. On the far left there are no matches, so it has a perfectly black value, as it goes to the right and starts to overlap it gets a gray value, indicating a partial match at some of the pixels of the template itself. Then when it perfectly over laps the red triangle in the image, it gives a perfectly white value. As it continues unto the right, it will go back to black indicating that there was a no match. 

![image](https://user-images.githubusercontent.com/79074273/155468344-48ec57a8-27cc-418d-8c30-c9989ab4e055.png)

![image](https://user-images.githubusercontent.com/79074273/155468497-1224043b-2b03-4a84-9b8c-cc3de8c8aa14.png)


- Typically, template matching is actually applied in a two-dimensional format but the concept is the same, your source template image will scroll horizontally and vertically across the entire image, taking a difference at each location. The sum result of that difference is put into the pixel value, where a zero sum difference mean the exact same images becomes white and a perfect difference become black. Typically you'll find there's lots of gray in your image as there are always going to be partial matches of some of the pixels in the template versus your image. 
- The example shown here, we have a yellow ball used as the template. Of course it's only one yellow ball in the actual scene and we can see very clearly in the output that the brightest spot of the image is exactly where we expect the ball to be. However there are partial matches elsewhere in the image most likely where the yellow channels seem to overlap. 

![image](https://user-images.githubusercontent.com/79074273/155468455-4e8c9143-73b9-415b-bac7-a62fbef20c0a.png)

- Typically with template matching you don't actually use an element from the source image yourself but something that is predetermined, such as a face or a known generic object that is expected to be found in the scene. 
- Given that, it's important to understand a few of the limitations of template matching. 
  - If your template is `scaled` compared to your Source image then it will `not work` very well
  - likewise if your template is `rotated` and the template looks different at those different rotations, it may `reduce the effectiveness` of the template matching.
  - Despite that it's still a very efficient algorithm and can be very useful in some scenarios.
 
 
   ![image](https://user-images.githubusercontent.com/79074273/155468672-eb7c56c8-8437-40e5-8d28-2ab2052bc3f3.png)
   
#### Template matching Application
- `result  =  cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED) ` method used to match the template and original image
- `min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)` method gives the min and max values and thier locations
```
import numpy as np
import cv2

img = cv2.imread("players.jpg",0)
cv2.imshow("original image",img)
template  = cv2.imread("template.jpg",0)
cv2.imshow("template",template)


result  =  cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)

min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)

print(max_val,max_loc)

cv2.circle(result,max_loc,15,255,4)

cv2.imshow("matching",result)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
result: 
![image](https://user-images.githubusercontent.com/79074273/155471509-5611a91c-4c35-4391-b164-ff1285611bff.png)

### 3.3 Face detection
#### Haar cascading
- `Haar cascade classifiers`, a form of future-based machine learning. It works by first training a classifier with set of labeled positive and negatives. Or in other words, indicating to the classifier that these are sets of images that have faces and these are sets of images that don't have faces. 
- This classifier then learns from the set by understanding and extracting features from all the images. 
- For example, it may naturally learn that the region of the eye is as typically darker than the region of the cheeks below and may use that as one of its thousands of indicators that help understand whether not a particular `region of interest`, or `ROI`, is a face or not. 
- After the training is completed, and a classifier is defined, we use the classifier in a cascaded manner to run through all the feature checks. This cascade method works like a waterfall, where you apply the fastest and most general checks first in order to quickly rule out areas that are definitely not matching a face without spending too much computational time. As it becomes more refined and goes through more classifiers, it gets more and more sure that the region of interest is actually a face. If it gets through all the cascaded classifiers, it is then marked as a valid face and outputs the bounding blocks. When we run the face detection algorithm and open CV, using the training data, we essentially leverage the already trained information into a cascade classifier which would then output the set of found faces and the regions of interests. 
- Note however, is not always perfect. And is possible that there will still be false positives and false negatives. Since your training data is rarely ever exactly the same as the applied data, you always are at risk at false negatives or positives. But there are parameters to tweak the classifier to make it more accurate for the particular situation.
 - `eye_cascade = cv2.CascadeClassifier(path)` method used to form a classifier, need to provide the xml file contains the trained data. [see the trained data xml](https://github.com/subrahmanyam-pampana/python-projects/blob/8f1fe4f77285d29831685cc8d9c070e14fde9a07/Ex_Files_OpenCV_Python_Dev/Ex_Files_OpenCV_Python_Dev/Exercise%20Files/Ch04/04_05%20Begin/haarcascade_frontalface_default.xml)
 -  `eyes = eye_cascade.detectMultiScale(gray,scaleFactor=1.02,minNeighbors=20,minSize = (10,10))` method return all the matched objects contains [x,y,w,h] attribues  
 ```
import numpy as np
import cv2

img = cv2.imread("crowd.jpg",1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
path = "haarcascade_frontalface_default.xml"

face_cascade  = cv2.CascadeClassifier(path)

faces = face_cascade.detectMultiScale(gray,scaleFactor=1.08,minNeighbors=7,minSize=(40,40) )

print(len(faces))
for (x,y,w,h) in faces:
  cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)

cv2.imshow("faces",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 ```
 result:
 
 ![image](https://user-images.githubusercontent.com/79074273/155477112-7d6e6449-5c9c-40f9-a96b-d22705094422.png)

### 3.4 Eye Detection
- [trained data xml](https://github.com/subrahmanyam-pampana/python-projects/blob/8f1fe4f77285d29831685cc8d9c070e14fde9a07/Ex_Files_OpenCV_Python_Dev/Ex_Files_OpenCV_Python_Dev/Exercise%20Files/Ch04/04_06%20Begin/haarcascade_eye.xml)
 ```
import numpy as np
import cv2


img = cv2.imread("faces.jpeg",1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
path = "haarcascade_eye.xml"

eye_cascade = cv2.CascadeClassifier(path)

eyes = eye_cascade.detectMultiScale(gray,scaleFactor=1.02,minNeighbors=20,minSize = (10,10))

for (x,y,w,h) in eyes:
  cx = int((x+x+w)//2) 
  cy = int((y+y+h)//2) 
  radius = int(w//2) 

  cv2.circle(img,(cx,cy),radius,(255,0,0),2)

cv2.imshow("Eyes",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 ```
 ## 4.Other Techniques and Applications
 - From a computer vision standpoint, we have only scratched the surface with the topics covered so far. Let's take a moment to briefly look at some other algorithms in the field. 
 - One of those applications we've already seen briefly under the hood is `machine learning`. Specifically, we have been looking at `supervised machine learning`. This is a form of machine-based learning where you train a classifier using already-tagged or identified data. For example, you start a pool of images that is an apple and then a pool of images which are not an apple. The classifier then builds little tests, extracting features from the image. And for each of those tests, it is evaluated of how well it indicates it being one image object or another. 
 
 ![image](https://user-images.githubusercontent.com/79074273/155656680-83e50b66-e7e9-43f4-a5dd-e1c006658942.png)
 
 ![image](https://user-images.githubusercontent.com/79074273/155656731-91aba736-ec05-497a-a266-98d1ff94ac6d.png)


   - When it comes to supervised machine learning, an important concept is the `confusion matrix`. The idea is that you can evaluate the effectiveness of your classifier or machine learning data by testing it against a set of images that were not used in the training process. Here you can see that the true positives are the diagonals and the false positives are all the other areas such as when a key is accidentally recognized as an apple. 
- Another area of computer vision is `text recognition`. I'm sure you could imagine countless applications for the usefulness of OCR text recognition. But to name a few explicitly, you can imagine enabling autonomous vehicles to read signs or being able to help auto-sort and process letters with handwritten addresses. It has a general process of identifying if text is present in an image such as looking for areas of high contrast or certain identified shapes, and then it will often segment and warp and then run a machine-learned process to extract features and do character recognition. Then, there could be further processing done to detect groupings of words, lines, or sentences, and so forth. 

![image](https://user-images.githubusercontent.com/79074273/155656815-0a8a54ba-f264-451e-a43d-6f156bb06ef1.png)

![image](https://user-images.githubusercontent.com/79074273/155656868-c5827769-326e-4a9e-babf-f54db55fc095.png)

 - Another field is `optical flow and object tracking` as well as `scene reconstruction`. Optical flow is a key algorithm and technique for real time applications as well as for other applications that are used to reconstruct a scene. The idea is to calculate and understand the apparent motion within a scene and for all the objects present in an image. This is done by evaluating the change of pixels over time and between frames. 
 - When it comes to `object tracking`, it is typically faster to track an object between frames than it is to detect an object between frames. 
  - For example, with face tracking, it may take some computational energy to detect whether or not a face is in the scene. But once you have found a face and you know a little bit about what that particular face looks like, it becomes easier to track it than it would be to detect that face on every frame of a video.
  ![image](https://user-images.githubusercontent.com/79074273/155656921-ee4453b9-bf89-418a-a901-7a86dc61cc2e.png) 
- Yet another niche but good example of the use case of computer vision technology is `the reading and creating of QR codes` or scanners in general. 
  - We can see here a typical flowchart for the QR reading algorithm. You might notice this pattern of starting with `detection`, then `segmentation` and `transformation` to put the image in a more consumable format. 
  
  ![image](https://user-images.githubusercontent.com/79074273/155656985-1381d308-063b-4f47-bd7f-49dd402452b4.png)
  
  - Detection could be done using something such as a `gradient frequency` or `edge detection map` to filter a range of patterns. Then, the QR code is corner-pin warped into a more parallel format where it then becomes easy to read off the individual bits of the QR code by reading the black versus white areas in a serial manner. 
  - Then, using an understanding of the format and structure of a QR code, the information is converted and extracted from the image. 
  - When it comes to generating a QR code, it's essentially the same process but in reverse. QR codes even have the ability to have built-in error or bit checking to help identify when a wrong character was detected due to, for example, bad segmentation or uneven lighting. Again, this is only a short sampling of the possibilities of this library, and new techniques are always being developed. For more use cases and resources, take a look at opencv.org
  
  ![image](https://user-images.githubusercontent.com/79074273/155657044-81e4411c-2cb7-42b0-9ca7-41016fe4f20f.png)

## 5.Next Steps

![image](https://user-images.githubusercontent.com/79074273/155657544-2a9275a0-ac74-4cfa-a4ff-56cb5df20d13.png)







