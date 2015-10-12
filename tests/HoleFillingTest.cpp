#include "opencv2/ts.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../src/HoleFilling.h"
#include "../src/util.h"
#include <iostream>

using cv::ellipse;
using cv::imread;
using cv::Mat;
using cv::Point;
using cv::randu;
using cv::Rect;
using cv::Scalar;
using cv::Size;
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

TEST(hole_filling_test, square_hole_on_repeated_texture_should_give_good_result)
{
    Mat img = imread("test_images/brick_pavement.jpg");
    convert_for_computation(img, 0.5f);

    // Add some hole
    Mat hole_mask = Mat::zeros(img.size(), CV_8U);

    hole_mask(Rect(70, 65, 10, 10)) = 1;
    int patch_size = 7;
    HoleFilling hf(img, hole_mask, patch_size);

    // Dump image with hole as black region.
    Mat img_with_hole_bgr;
    cvtColor(img, img_with_hole_bgr, CV_Lab2BGR);
    img_with_hole_bgr.setTo(Scalar(0,0,0), hole_mask);
    imwrite("brick_pavement_hole.exr", img_with_hole_bgr);

    // Dump reconstructed image
    Mat filled = hf.run();
    cvtColor(filled, filled, CV_Lab2BGR);
    imwrite("brick_pavement_hole_filled.exr", filled);


    // The reconstructed image should be close to the original one, in this very simple case.
    Mat img_bgr;
    cvtColor(img, img_bgr, CV_Lab2BGR);
    double ssd = pmutil::ssd(img_bgr, filled);
    double mse = ssd / (img_bgr.cols * img_bgr.rows);
    EXPECT_LT(mse, 1e-4);
}

TEST(hole_filling_test, rectangular_hole_on_repeated_texture_should_give_good_result)
{
    Mat img = imread("test_images/brick_pavement.jpg");
    convert_for_computation(img, 0.5f);

    // Add some hole
    Mat hole_mask = Mat::zeros(img.size(), CV_8U);

    hole_mask(Rect(70, 65, 5, 20)) = 1;
    int patch_size = 7;
    HoleFilling hf(img, hole_mask, patch_size);

    // Dump image with hole as black region.
    Mat img_with_hole_bgr;
    cvtColor(img, img_with_hole_bgr, CV_Lab2BGR);
    img_with_hole_bgr.setTo(Scalar(0,0,0), hole_mask);
    imwrite("brick_pavement_hole.exr", img_with_hole_bgr);

    // Dump reconstructed image
    Mat filled = hf.run();
    cvtColor(filled, filled, CV_Lab2BGR);
    imwrite("brick_pavement_hole_filled.exr", filled);


    // The reconstructed image should be close to the original one, in this very simple case.
    Mat img_bgr;
    cvtColor(img, img_bgr, CV_Lab2BGR);
    double ssd = pmutil::ssd(img_bgr, filled);
    double mse = ssd / (img_bgr.cols * img_bgr.rows);
    EXPECT_LT(mse, 1e-4);
}

TEST(hole_filling_test, elliptical_hole_on_repeated_texture_should_give_good_result)
{
    Mat img = imread("test_images/brick_pavement.jpg");
    convert_for_computation(img, 0.5f);

    // Add some hole
    Mat hole_mask = Mat::zeros(img.size(), CV_8U);

    Point center(100, 100);
    Size axis(20, 5);
    float angle = 20;
    ellipse(hole_mask, center, axis, angle, 0, 360, Scalar(1,1,1), -1);
    int patch_size = 7;
    HoleFilling hf(img, hole_mask, patch_size);

    // Dump image with hole as black region.
    Mat img_with_hole_bgr;
    cvtColor(img, img_with_hole_bgr, CV_Lab2BGR);
    img_with_hole_bgr.setTo(Scalar(0,0,0), hole_mask);
    imwrite("brick_pavement_hole.exr", img_with_hole_bgr);

    // Dump reconstructed image
    Mat filled = hf.run();
    cvtColor(filled, filled, CV_Lab2BGR);
    imwrite("brick_pavement_hole_filled.exr", filled);


    // The reconstructed image should be close to the original one, in this very simple case.
    Mat img_bgr;
    cvtColor(img, img_bgr, CV_Lab2BGR);
    double ssd = pmutil::ssd(img_bgr, filled);
    double mse = ssd / (img_bgr.cols * img_bgr.rows);
    // TODO: This does not give satisfying results.
    EXPECT_LT(mse, 1e-5);
}