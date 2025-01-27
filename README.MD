# Example Usage
darknet detector test trash.data trash.cfg ./content/backup/trash_best.weights ./../trash_ICRA19/dataset/test/obj0314_frame0000032.jpg

# Installation Requirements and Resources: 
The software and versions are as follows. 

    WSL on Ubuntu 22.04 

    C++ 17 or newer 

    CMake 3.30.2 

    OpenCV v4.5.4 

    CUDA 12.6 

    CuDNN 9.6.0 

Requirements to build and use Darknet V3 for real time application, as well as using it to train models. CUDA and CuDNN require a compatible NVIDIA GPU.  
Follow these instructions to build darknet: https://github.com/hank-ai/darknet
