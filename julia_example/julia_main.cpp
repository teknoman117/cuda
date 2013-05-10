#include <iostream>
#include "julia.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define DIM 1024

int main (void)
{
    // Create an image we can use
    cv::Mat cpu_bitmap (DIM, DIM, CV_8UC4, cv::Scalar(100));
    
    // Create a window to display our resulting code with
    cv::namedWindow("Julia Set");
    
    // Do our GPU calculations here
    julia_set(DIM, DIM, cpu_bitmap.data);
    
    // Show our image
    cv::imshow("Julia Set", cpu_bitmap);
    
    // Wait for key
    cv::waitKey();
    
    return 0;
}  