#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo.hpp"
#include "util.h"
#include <iostream>

using cv::getTickCount;
using cv::getTickFrequency;
using cv::imread;
using cv::seamlessClone;
using cv::Mat;
using cv::Point;
using std::cout;
using std::endl;

const float RESIZE_FACTOR = 0.5;

int main( int argc, char** argv )
{
    // Load image
    Mat src = imread(argv[1]);
    Mat dst = imread(argv[2]);
    if (!src.data || !dst.data) {
        printf("Need two pixtures as arguments.\n");
        return -1;
    }


    // Create an all white mask
    Mat src_mask = 255 * Mat::ones(src.rows, src.cols, src.depth());

    // The location of the center of the src in the dst
    Point center(dst.cols/2,dst.rows/2);

    // Seamlessly clone src into dst and put the results in output
    Mat normal_clone;
    Mat mixed_clone;

    seamlessClone(src, dst, src_mask, center, normal_clone, cv::NORMAL_CLONE);
    seamlessClone(src, dst, src_mask, center, mixed_clone, cv::MIXED_CLONE);

    // Save results
    imwrite("opencv-normal-clone-example.jpg", normal_clone);
    imwrite("opencv-mixed-clone-example.jpg", mixed_clone);

    /*
    if (argc == 3) {
        Mat original = imread(argv[2]);
        if (!original.data) {
            printf("Failed to read comparison image.\n");
            return -2;
        }
        resize(original, original, Size(), RESIZE_FACTOR, RESIZE_FACTOR);
        original.convertTo(original, CV_32FC3, 1 / 255.f);

        Mat filled_bgr;
        cvtColor(filled, filled_bgr, CV_Lab2BGR);

        cout << "SSD between original and hole filled version: " << ssd(original, filled_bgr) << endl;
        cout << "Time: " << toc << endl;
    }
     */
    return 0;
}