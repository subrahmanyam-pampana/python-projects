# IMAGE PROCESSING USING PYTHON OPEN CV LIBRARY
## INSTALLATION OF NUMPY AND OPEN CV
1. First install `pipwin` if it not installed already. it is the offcial installer for installing `numpy` or any other packeages from unofficial location [unoffcila py libs](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
2. For installing pipwin use `pip install pipwin`
3. After installing pipwin install numpy library using `pipwin install numpy`. it Installs the latest version of numpy compatble [numpy lib](https://download.lfd.uci.edu/pythonlibs/x6hvwk7i/numpy-1.22.2+mkl-cp310-cp310-win_amd64.whl) with currect version of python.
4. For testing the installed packages, run the below commands in cmd promt one by one. if packages correctly installed, it will return version
  ```
  import numpy
  numpy.__version__
  
  import cv2
  cv2.__version__
  ```
## BASICS OF OPEN CV(Computer vision)
### Reading and Writing the Image
```
import numpy as np
import cv2
#read the image
img = cv2.imread("myImagePath.png")

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
```






