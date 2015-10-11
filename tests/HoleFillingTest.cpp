#include "opencv2/ts.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../src/HoleFilling.h"
#include "../src/util.h"

using cv::imread;
using cv::Mat;
using cv::Point;
using cv::randu;
using cv::Rect;
using cv::Scalar;
using cv::Vec3f;
using pmutil::convert_for_computation;

TEST(hole_filling_test, square_hole_on_random_image_should_produce_correct_target_rect)
{
    // Make some random image data.
    Mat img = Mat(100, 100, CV_32FC1);
    randu(img, 0.f, 1.f);

    // Put a hole in middle.
    Mat hole = Mat::zeros(100, 100, CV_8U);

    hole(Rect(50, 50, 10, 10)) = 1;
    int patch_size = 7;
    HoleFilling hf(img, hole, patch_size);

    Rect expected_target_rect(Point(50 - 6, 50 - 6), Point(60 + 6, 60 + 6));

    ASSERT_EQ(expected_target_rect, hf.computeTargetRect(img, hole, patch_size));
}

TEST(hole_filling_test, initial_guess_should_make_sense)
{
    Mat img = imread("test_images/gitter.jpg");
    convert_for_computation(img, 1.f);

    // Add some hole
    Mat hole = Mat::zeros(img.size(), CV_8U);

    hole(Rect(65, 70, 10, 10)) = 1;
    int patch_size = 7;
    HoleFilling hf(img, hole, patch_size);

    Mat bgr;
    cvtColor(hf._target_area_pyr[hf._nr_scales], bgr, CV_Lab2BGR);
    imwrite("gitter_hole_initialized.exr", bgr);
    int i = 0;
    for (Mat m: hf._hole_pyr) {
        imwrite("hole" + std::to_string(i) + ".exr", m);
        i++;
    }
    // TODO: Test something sensible.
}

TEST(hole_filling_test, square_hole_on_random_image)
{
    Mat img = imread("test_images/gitter.jpg");
    convert_for_computation(img, 1.f);

    // Add some hole
    Mat hole = Mat::zeros(img.size(), CV_8U);

    hole(Rect(65, 70, 10, 10)) = 1;
    int patch_size = 7;
    HoleFilling hf(img, hole, patch_size);

    Mat filled = hf.run();

    Mat bgr;
    cvtColor(filled, bgr, CV_Lab2BGR);
    imwrite("gitter_hole_filled.exr", filled);

    hf._target_area_pyr[hf._nr_scales];
    // TODO: Test something sensible.
}
