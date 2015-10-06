#include <iostream>

#include "opencv2/ts.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../src/RandomizedPatchMatch.h"

using cv::Mat;

const double EPSILON = 1e-6;
TEST(randomized_patch_match_test, on_two_identical_trivial_images)
{
    Mat img1 = Mat::ones(20, 20, CV_32FC1);
    Mat img2 = Mat::ones(20, 20, CV_32FC1);

    RandomizedPatchMatch rpm(img1, img2, 7);
    Mat diff = rpm.match();
    double overall_ssd = sum(diff)[2];

    ASSERT_NEAR(0.0, overall_ssd, EPSILON);
}

TEST(randomized_patch_match_test, on_two_very_different_trivial_images)
{
    Mat img1 = Mat::zeros(20, 20, CV_32FC1);
    Mat img2 = Mat::ones(20, 20, CV_32FC1);

    RandomizedPatchMatch rpm(img1, img2, 7);
    Mat diff = rpm.match();
    double overall_ssd = sum(diff)[2];

    // We expect for every patch (size - patch_size)^2 the maximum deviation of 7*7 (every pixel has SSD 1)
    double expected_ssd = (20 - 7) * (20 - 7) * 7 * 7;
    ASSERT_NEAR(expected_ssd, overall_ssd, EPSILON);
}