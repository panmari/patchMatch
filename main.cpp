#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ExhaustivePatchMatch.h"
#include "RandomizedPatchMatch.h"

#include <iostream>

using cv::Mat;
using cv::Size;
using cv::namedWindow;
using cv::imread;
using std::cout;
using std::endl;
using cv::String;
using cv::split;

String result_window = "Result window";

int main( int argc, char** argv )
{
    /// Load image and template
    Mat img = imread( argv[1], 1 );
    Mat img2 = imread( argv[2], 1);

    if (!img.data or !img2.data) {
        printf("Need two pictures as arguments.\n");
        return -1;
    }

    // For fast testing, make it tiny
    float resizeFactor = 0.5;
    resize(img, img, Size(), resizeFactor, resizeFactor);
    resize(img2, img2, Size(), resizeFactor, resizeFactor);
    // Convert images to lab, default returned by imread is BGR.
    cvtColor(img, img, CV_BGR2Lab);
    cvtColor(img2, img2, CV_BGR2Lab);

    cout << "Size of img1: " << img.size() << endl;
    cout << "Size of img2: " << img2.size() << endl;

    RandomizedPatchMatch rpm(img, img2);

    ExhaustivePatchMatch epm(img, img2);

    /// Create windows
    namedWindow( result_window, CV_WINDOW_AUTOSIZE );

    Mat minDistImg = rpm.match(7);
    //Mat minDistImgExhaustive = epm.match(7);
    // Normalize and show
    //normalize(minDistImg, minDistImg, 0, 1, cv::NORM_MINMAX, CV_32FC1, Mat() );
    //imshow(result_window, minDistImg);
    // Convert and save to disk.
    //minDistImg.convertTo(minDistImg, CV_16U, 255*255);
    Mat xoffsets, yoffsets, diff;
    Mat out[] = {xoffsets, yoffsets, diff};
    split(minDistImg, out);

    normalize(out[0], out[0], 0, 1, cv::NORM_MINMAX, CV_32FC1, Mat() );
    imwrite("xoffsets.exr", out[0]);
    normalize(out[1], out[1], 0, 1, cv::NORM_MINMAX, CV_32FC1, Mat() );
    imwrite("yoffsets.exr", out[1]);
    imwrite("minDistImg.exr", out[2]);

    cv::waitKey(0);
    return 0;
}
