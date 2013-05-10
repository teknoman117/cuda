By Nathaniel Lewis

Do whatever you please with the code I really don't care at all.  If you wish (it would be nice) to just put reference to who wrote this just link back to my video

This code depends on 
 - G++
 - OpenCV 2.3+
 - CUDA 4.0+ sdk

This code should work on any Unix like system (Mac OS X and Linux) as long as you have the nvcc compiler and g++ compiler in your binary path and OpenCV 2.3 installed in a place with default paths (/usr on Ubuntu, /usr/local on a Mac).  You MUST add nvcc to PATH on a mac.  Just put in your .bash_profile: 
export PATH=$PATH:<PATH TO THE DIRECTORY CONTAINING nvcc without these brackets>