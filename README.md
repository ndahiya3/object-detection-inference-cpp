# object-detection-inference-cpp
Demo to run object detection using a trained model using Tensorflow C++ API
------------------------------------------------------------------------------------------

1) Objective: To develop C++ code to run obejct detection on a trained model using Tensorflow

This demo is based on an object detection tutorial available here:
https://www.youtube.com/watch?v=rWFg6R5ccOc

The corresponding project repository and instructions available here:
https://github.com/MicrocontrollersAndMore/TensorFlow_Tut_3_Object_Detection_Walk-through

The project uses python/opencv/tensorflow object_detection modules to retrain an inception v2
model to detect traffic lights. Training images are also provided by the author.

-----------------------------------------------------------------------------------------
2) Main steps followed:

a) The instructions in the original tutorial documents referenced above were followed by
using the provided python code to generate a retrained inception v2 model to detect traffic
lights using 117 training images and 10 testing images. Any modifications needed such as
file paths etc were done and any dependencies installed. Finally a trained model was generated
used for detection of traffic lights in images.

b) The final step in the object detection pipeline is running the trained inference graph on
testing images and visualizing the detection results. This part of the code was redone in
C++ using Tensorflow C++ API and opencv and is the main contribution in this project.

-----------------------------------------------------------------------------------------

3) Detailed description:

There are two ways to run inference in C++ using tensorflow. First is to have the C++ file
in tensorflow examples directory and using bazel to build and run the code. The main problem
with that is it doesn't play well with OpenCV and tensorflow has limited image manipulation
capabilities needed to visulize object detection results. Hence the second option is to
compile your C++ program as a standalone program linking against tensorflow and opencv
shared libraries.

Consequently the second approach was followed in this project. This issue has document in
several places:
https://github.com/tensorflow/models/issues/1741
https://github.com/tensorflow/tensorflow/issues/2412

The steps used to compile standalone tensorflow libraries were mainly followed from here
without many issues:
https://tuanphuc.github.io/update-on-standalone-tensorflow-cpp/
https://tuanphuc.github.io/standalone-tensorflow-cpp/

The main code is in file 5_test.cpp. The main inspirations for the code come from the
followwing sources:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/multibox_detector/main.cc
https://github.com/lysukhin/tensorflow-object-detection-cpp

Other main resources include:

https://www.tensorflow.org/api_guides/cc/guide for Tensorflow C++ API
https://developers.google.com/protocol-buffers/docs/cpptutorial for google protocal buffers
usage.

------------------------------------------------------------------------------------------
4) Files included:

The object detection tutorial walkthrough source code files are included in here for convenience.

The training and test images along with any other files generated by the python pipeline such as tfr records,
trained inference graph etc. are available at the following location:

https://drive.google.com/open?id=1rh0RWcsGTwHTN4We0CuKU86LS2k8-ydn

The cpp code including the files needed to compile standalone program as per instructions in 
https://tuanphuc.github.io/standalone-tensorflow-cpp/ are also included.

Finally the compiled libraries and main program are also included in test-object-detect-cpp
subdirectory. Results of c++ program inference run are also provided in output_images sub-sub
directory.

------------------------------------------------------------------------------------------
5) Instructions to run the program:
main steps to run CPP part of the code are:

Adjust the paths/variables in Makefile as needed.

The first step would be to try:

make run

as the compiled binary is provided (linux x86_64 system). If that fails then:

make clean
make
make run

You'll have to satisfy the dependencies listed in the Makefile otherwise follow one or more
steps from section 3.

------------------------------------------------------------------------------------------
6) Dependencies:
Standalone compilation of tensorflow + opencv requires the exact same protobuf library version
used to compile tensorflow from sources. This project compiled Tensorflow v 1.8.0 with Protobuf v 3.5.0
Opencv == 2.4.9
