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
    *  `img.shape` gives `(Rows,columns,num_channels)`
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
### Pixel manupulation and Filtering 
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
  
 








