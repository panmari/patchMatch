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

/**
 * Tries to reconstruct the second image with patches from the first images.
 */
int main( int argc, char** argv )
{
    /// Load image and template
    Mat source = imread( argv[1], 1 );

    if (!source.data) {
        printf("Need a picture with a magenta region as arguments.\n");
        return -1;
    }
    // For fast testing, make it tiny
    convert_for_computation(source, RESIZE_FACTOR);


    return 0;
}