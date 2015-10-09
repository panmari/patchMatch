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

    ASSERT_EQ(expected_target_rect, hf._target_rect);
}

TEST(hole_filling_test, square_hole_on_random_image)
{
    // Make some random image data.
    Mat img = imread("test_images/gitter.jpg");
    convert_for_computation(img, 1.f);

    // Add some hole
    Mat hole = Mat::zeros(img.size(), CV_8U);

    hole(Rect(50, 50, 10, 10)) = 1;
    int patch_size = 7;
    HoleFilling hf(img, hole, patch_size);

    imwrite("test_hole_filling.exr", hf._target_area);
}
