#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ExhaustivePatchMatch.h"

#include <iostream>

using cv::Mat;
using cv::Size;
using cv::namedWindow;
using cv::imread;
using std::cout;
using std::endl;
using cv::String;

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

    cout << "Size of img1: " << img.size() << endl;
    cout << "Size of img2: " << img2.size() << endl;

    ExhaustivePatchMatch epm(img, img2);

    /// Create windows
    namedWindow( result_window, CV_WINDOW_AUTOSIZE );

    Mat minDistImg = epm.match(7);
    // Normalize and show
    normalize(minDistImg, minDistImg, 0, 1, cv::NORM_MINMAX, CV_32FC1, Mat() );
    imshow(result_window, minDistImg);
    // Convert and save to disk.
    minDistImg.convertTo(minDistImg, CV_16U, 255*255);
    imwrite("minDistImg.png", minDistImg);


    cv::waitKey(0);
    return 0;
}
