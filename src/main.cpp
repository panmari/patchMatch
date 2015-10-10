#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ExhaustivePatchMatch.h"
#include "RandomizedPatchMatch.h"
#include "TrivialReconstruction.h"
#include "VotedReconstruction.h"
#include "util.h"

#include <iostream>

using cv::addWeighted;
using cv::Mat;
using cv::Size;
using cv::namedWindow;
using cv::imread;
using cv::Rect;
using cv::Scalar;
using cv::String;
using cv::split;
using pmutil::convert_for_computation;
using pmutil::ssd;
using std::cout;
using std::endl;

const float RESIZE_FACTOR = 0.5;
const int PATCH_SIZE = 7;
void convert_for_computation(Mat &img);

int main( int argc, char** argv )
{
    /// Load image and template
    Mat img = imread( argv[1], 1 );
    Mat img2 = imread( argv[2], 1);

    if (!img.data || !img2.data) {
        printf("Need two pictures as arguments.\n");
        return -1;
    }
    // For later comparison.
    Mat original;
    resize(img, original, Size(), RESIZE_FACTOR, RESIZE_FACTOR);
    original.convertTo(original, CV_32FC3, 1 / 255.f);

    // For fast testing, make it tiny
    convert_for_computation(img, RESIZE_FACTOR);
    convert_for_computation(img2, RESIZE_FACTOR);

    cout << "Size of img1: " << img.size() << endl;
    cout << "Size of img2: " << img2.size() << endl;

    RandomizedPatchMatch rpm(img, img2, PATCH_SIZE);

    ExhaustivePatchMatch epm(img, img2);

    Mat minDistImg = rpm.match();
    //Mat minDistImg = epm.match(PATCH_SIZE);

    TrivialReconstruction tr(minDistImg, img2, PATCH_SIZE);
    Mat reconstructed = tr.reconstruct();
    imwrite("reconstructed.exr", reconstructed);

    VotedReconstruction vr(minDistImg, img2, PATCH_SIZE);
    Mat reconstructed2 = vr.reconstruct();
    imwrite("reconstructed_voted.exr", reconstructed2);

    cout << "SSD trivial reconstruction: " << ssd(reconstructed, original) << endl;
    cout << "SSD voted reconstruction: " << ssd(reconstructed2, original) << endl;
    return 0;
}