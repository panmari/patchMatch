#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "HoleFilling.h"
#include "util.h"

#include <iostream>

using cv::imread;
using cv::inRange;
using cv::Mat;
using cv::resize;
using cv::Scalar;
using cv::Size;
using pmutil::convert_for_computation;
using pmutil::ssd;
using pmutil::imwrite_lab;
using std::cout;
using std::endl;

const float RESIZE_FACTOR = 0.5;
const int PATCH_SIZE = 7;

/**
 * Takes one image with a 'hole region' (pixels in magenta) as input. The hole region will then be inpainted.
 * If a second image is given as argument, the ssd between the reconstructed one and this one will be printed to stdout.
 */
int main( int argc, char** argv )
{
    // Load image
    Mat source = imread( argv[1]);

    if (!source.data) {
        printf("Need a picture with a magenta region as arguments.\n");
        return -1;
    }
    // Pixels of the color magenta are treated as hole.
    Scalar hole_color = Scalar(255, 0, 255);
    Mat hole_mask;
    inRange(source, hole_color, hole_color, hole_mask);
    if (RESIZE_FACTOR != 1.f) {
        resize(hole_mask, hole_mask, Size(), RESIZE_FACTOR, RESIZE_FACTOR);
    }    // For fast testing, make it tiny
    convert_for_computation(source, RESIZE_FACTOR);

    HoleFilling hf(source, hole_mask, PATCH_SIZE);
    Mat filled = hf.run();

    imwrite_lab("result.exr", filled);

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
    }
    return 0;
}