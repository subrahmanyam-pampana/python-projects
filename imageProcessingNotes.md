# IMAGE PROCESSING USING PYTHON OPEN CV LIBRARY
## INSTALLATION OF NUMPY AND OPEN CV
1. First install `pipwin` if it not installed already. it is the offcial installer for installing `numpy` or any other packeages from unofficial location [unoffcila py libs](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
2. For installing pipwin use `pip install pipwin`
3. After installing pipwin install numpy library using `pipwin install numpy`. it Installs the latest version of numpy compatble [numpy lib](https://download.lfd.uci.edu/pythonlibs/x6hvwk7i/numpy-1.22.2+mkl-cp310-cp310-win_amd64.whl) with currect version of python.
4. Install Open cv library by running `pipwin install opencv-python`. or download it from [open cv](https://download.lfd.uci.edu/pythonlibs/x6hvwk7i/opencv_python-4.5.5+mkl-cp310-cp310-win_amd64.whl)
5. For testing the installed packages, run the below commands in cmd promt one by one. if packages correctly installed, it will return version
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








