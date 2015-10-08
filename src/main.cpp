#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ExhaustivePatchMatch.h"
#include "RandomizedPatchMatch.h"
#include "TrivialReconstruction.h"
#include "VotedReconstruction.h"

#include <iostream>

using cv::addWeighted;
using cv::Mat;
using cv::Size;
using cv::namedWindow;
using cv::imread;
using std::cout;
using std::endl;
using cv::Scalar;
using cv::String;
using cv::split;

String result_window = "Result window";
void convert_for_computation(Mat &img);

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
    convert_for_computation(img);
    convert_for_computation(img2);

    cout << "Size of img1: " << img.size() << endl;
    cout << "Size of img2: " << img2.size() << endl;

    RandomizedPatchMatch rpm(img, img2, 7);

    ExhaustivePatchMatch epm(img, img2);

    Mat minDistImg = rpm.match();
    //Mat minDistImg = epm.match(7);

    TrivialReconstruction tr(minDistImg, img2);
    Mat reconstructed = tr.reconstruct();
    imwrite("reconstructed.exr", reconstructed);

    VotedReconstruction vr(minDistImg, img2, 7);
    Mat reconstructed2 = vr.reconstruct();
    imwrite("reconstructed_voted.exr", reconstructed2);
    Mat diff;
    addWeighted(img, 1, reconstructed2, -1, 0, diff);
    Mat diff_sqr = diff.mul(diff);
    Scalar per_channel_diff = sum(diff_sqr);
    cout << per_channel_diff[0] + per_channel_diff[1] + per_channel_diff[2] << endl;
    return 0;
}

// Convert images to lab, default returned by imread is BGR.
void convert_for_computation(Mat &img) {
    const float resizeFactor = 0.5;
    resize(img, img, Size(), resizeFactor, resizeFactor);
    img.convertTo(img, CV_32FC3, 1 / 255.f);
    cvtColor(img, img, CV_BGR2Lab);
}