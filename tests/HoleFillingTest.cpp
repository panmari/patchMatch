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

    ASSERT_EQ(expected_target_rect, hf._target_rect_pyr[0]);
}

TEST(hole_filling_test, initial_guess_should_make_sense_for_grid_image)
{
    Mat img = imread("test_images/gitter.jpg");
    convert_for_computation(img, 1.f);

    // Add some hole
    Mat hole = Mat::zeros(img.size(), CV_8U);

    hole(Rect(65, 70, 10, 10)) = 1;
    int patch_size = 7;
    HoleFilling hf(img, hole, patch_size);

    Mat initial_guess = hf._target_area_pyr[hf._nr_scales];
    // TODO: Make some assertions.

    Mat bgr;
//    cvtColor(initial_guess, bgr, CV_Lab2BGR);
//    imwrite("gitter_hole_initialized.exr", bgr);
}

TEST(hole_filling_test, square_hole_on_grid_image)
{
    Mat img = imread("test_images/brick_pavement.jpg");
    convert_for_computation(img, 1.f);

    // Add some hole
    Mat hole_mask = Mat::zeros(img.size(), CV_8U);

    hole_mask(Rect(70, 65, 20, 20)) = 1;
    int patch_size = 7;
    HoleFilling hf(img, hole_mask, patch_size);

    // Dump image with hole
    Mat img_with_hole_bgr;
    cvtColor(img, img_with_hole_bgr, CV_Lab2BGR);
    img_with_hole_bgr.setTo(Scalar(0,0,0), hole_mask);
    imwrite("gitter_hole.exr", img_with_hole_bgr);

//    Mat filled_with_initial_guess = hf.solutionFor(hf._nr_scales);
//    cvtColor(filled_with_initial_guess, bgr, CV_Lab2BGR);
//    imwrite("gitter_hole_filled_initial_guess.exr", bgr);
    Mat bgr;
    Mat filled = hf.run();
    cvtColor(filled, bgr, CV_Lab2BGR);
    imwrite("gitter_hole_filled.exr", bgr);

    hf._target_area_pyr[hf._nr_scales];
    // TODO: Test something sensible.
}

// TODO: Test making hole by combining multiple rectangles to non-rectangular form.