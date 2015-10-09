#include "opencv2/ts.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../src/HoleFilling.h"

using cv::Mat;
using cv::Point;
using cv::randu;
using cv::Rect;
using cv::Scalar;
using cv::Vec3f;

TEST(hole_filling_test, square_hole_on_random_image)
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
