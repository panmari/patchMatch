#include "opencv2/ts.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../src/RandomizedPatchMatch.h"

using cv::Mat;
using cv::Scalar;
using cv::Vec3f;

const double EPSILON = 1e-6;
TEST(randomized_patch_match_test, on_two_identical_trivial_images)
{
    Mat img1 = Mat::ones(20, 20, CV_32FC1);
    Mat img2 = Mat::ones(20, 20, CV_32FC1);

    RandomizedPatchMatch rpm(img1, img2, 7);
    Mat diff = rpm.match();
    Scalar diff_sums = sum(diff);
    double overall_ssd = diff_sums[2];

    ASSERT_NEAR(0.0, overall_ssd, EPSILON);
}

TEST(randomized_patch_match_test, on_two_very_different_trivial_images)
{
    Mat img1 = Mat::zeros(20, 20, CV_32FC1);
    Mat img2 = Mat::ones(20, 20, CV_32FC1);

    RandomizedPatchMatch rpm(img1, img2, 7);
    Mat diff = rpm.match();
    Scalar diff_sums = sum(diff);
    double overall_ssd = diff_sums[2];

    // We expect for every patch (size - patch_size)^2 the maximum deviation of 7*7 (every pixel has SSD 1)
    double expected_ssd = (20 - 7) * (20 - 7) * 7 * 7;
    ASSERT_NEAR(expected_ssd, overall_ssd, EPSILON);
}

TEST(randomized_patch_match_test, all_offsets_inside_image_on_random_images)
{
    Mat img1 = Mat::zeros(20, 20, CV_32FC1);
    randu(img1, Scalar::all(0.0), Scalar::all(1.0f));
    Mat img2 = Mat::ones(20, 20, CV_32FC1);
    randu(img2, Scalar::all(0.0), Scalar::all(1.0f));

    RandomizedPatchMatch rpm(img1, img2, 7);
    Mat diff = rpm.match();
    for(int x = 0; x < diff.cols; x++) {
        for(int y = 0; y < diff.rows; y++) {
            Vec3f d = diff.at<Vec3f>(y, x);
            int matching_patch_x = x + d[0];
            int matching_patch_y = y + d[1];
            ASSERT_GE(matching_patch_x, 0) << "Failed for offset in (" << x << "," << y << ")";
            ASSERT_GE(matching_patch_y, 0) << "Failed for offset in (" << x << "," << y << ")";
            ASSERT_LT(matching_patch_x, img2.cols) << "Failed for offset in (" << x << "," << y << ")";
            ASSERT_LT(matching_patch_y, img2.rows) << "Failed for offset in (" << x << "," << y << ")";
        }
    }
}

TEST(randomized_patch_match_test, all_offsets_inside_image_on_initialized_offsets)
{
    Mat img1 = Mat::zeros(20, 20, CV_32FC1);
    randu(img1, Scalar::all(0.0), Scalar::all(1.0f));
    Mat img2 = Mat::ones(20, 20, CV_32FC1);
    randu(img2, Scalar::all(0.0), Scalar::all(1.0f));

    RandomizedPatchMatch rpm(img1, img2, 7);
    Mat diff = rpm._offset_map;
    for(int x = 0; x < diff.cols; x++) {
        for(int y = 0; y < diff.rows; y++) {
            Vec3f d = diff.at<Vec3f>(y, x);
            int matching_patch_x = x + d[0];
            int matching_patch_y = y + d[1];
            ASSERT_GE(matching_patch_x, 0) << "Failed for offset in (" << x << "," << y << ")";
            ASSERT_GE(matching_patch_y, 0) << "Failed for offset in (" << x << "," << y << ")";
            ASSERT_LT(matching_patch_x, img2.cols) << "Failed for offset in (" << x << "," << y << ")";
            ASSERT_LT(matching_patch_y, img2.rows) << "Failed for offset in (" << x << "," << y << ")";
        }
    }
}