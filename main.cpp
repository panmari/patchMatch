#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ExhaustivePatchMatch.h"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

String result_window = "Result window";

/// Function Headers
void MatchingMethod(double *minVal, Point *minLoc);
string getImgType(int imgTypeInt);
void exhaustivePatchMatch(int);

/** @function main */
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

    ExhaustivePatchMatch epm(img, img2);

    /// Create windows
    namedWindow( result_window, CV_WINDOW_AUTOSIZE );

    Mat minDistImg = epm.match(7);
    // Normalize and show
    normalize(minDistImg, minDistImg, 0, 1, NORM_MINMAX, CV_32FC1, Mat() );
    imshow(result_window, minDistImg);
    // Convert and save to disk.
    minDistImg.convertTo(minDistImg, CV_16U, 255*255);
    imwrite("minDistImg.png", minDistImg);


    waitKey(0);
    return 0;
}

string getImgType(int imgTypeInt)
{
    int numImgTypes = 35; // 7 base types, with five channel options each (none or C1, ..., C4)

    int enum_ints[] =       {CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
                             CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
                             CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
                             CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
                             CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
                             CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
                             CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};

    string enum_strings[] = {"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
                             "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
                             "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
                             "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
                             "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
                             "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
                             "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};

    for(int i=0; i<numImgTypes; i++)
    {
        if(imgTypeInt == enum_ints[i]) return enum_strings[i];
    }
    return "unknown image type";
}
