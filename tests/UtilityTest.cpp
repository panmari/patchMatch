#include "opencv2/ts.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../src/util.h"

using cv::Mat;
using cv::Rect;
using cv::Vec3f;
using pmutil::naiveMeanShift;
using pmutil::createRotatedImages;
using std::vector;

TEST(utility_test, naive_mean_shift_with_only_one_color)
{
    vector<Vec3f> colors{Vec3f(20, 20, 20)};
    float sigma = 20;

    vector<Vec3f> modes;
    vector<int> mode_assignments;

    naiveMeanShift(colors, sigma, &modes, &mode_assignments);

    ASSERT_EQ(modes.size(), 1);
    ASSERT_EQ(mode_assignments.size(), 1);

    ASSERT_EQ(modes[0], colors[0]);
    ASSERT_EQ(mode_assignments[0], 0);
}

TEST(utility_test, naive_mean_shift_with_two_very_different_colors_and_small_kernel)
{
    vector<Vec3f> colors{Vec3f(20, 20, 20), Vec3f{50, 50, 50}};
    float sigma = 1;

    vector<Vec3f> modes;
    vector<int> mode_assignments;

    naiveMeanShift(colors, sigma, &modes, &mode_assignments);

    EXPECT_EQ(modes.size(), 2);
    EXPECT_EQ(mode_assignments.size(), 2);

    EXPECT_EQ(modes[0], colors[0]);
    EXPECT_EQ(mode_assignments[0], 0);
    EXPECT_EQ(modes[1], colors[1]);
    EXPECT_EQ(mode_assignments[1], 1);
}

TEST(utility_test, naive_mean_shift_with_two_very_different_colors_and_large_kernel)
{
    vector<Vec3f> colors{Vec3f(20, 20, 20), Vec3f{50, 50, 50}};
    float sigma = 6;

    vector<Vec3f> modes;
    vector<int> mode_assignments;

    naiveMeanShift(colors, sigma, &modes, &mode_assignments);

    EXPECT_EQ(modes.size(), 1);
    EXPECT_EQ(mode_assignments.size(), 2);

    Vec3f mode = modes[0];
    for (int c = 0; c < 3; c++) {
        EXPECT_NEAR(mode[c], 35, 0.01);
    }
    EXPECT_EQ(mode_assignments[0], 0);
    EXPECT_EQ(mode_assignments[1], 0);
}

TEST(utility_test, sigma_zero_should_work_for_only_one_color_given)
{
    vector<Vec3f> colors{Vec3f(20, 20, 20)};
    float sigma = 0;

    vector<Vec3f> modes;
    vector<int> mode_assignments;

    naiveMeanShift(colors, sigma, &modes, &mode_assignments);

    EXPECT_EQ(modes.size(), 1);
    EXPECT_EQ(mode_assignments.size(), 1);

    Vec3f mode = modes[0];
    for (int c = 0; c < 3; c++) {
        EXPECT_NEAR(mode[c], 20, 0.01);
    }
    EXPECT_EQ(mode_assignments[0], 0);
}

TEST(utility_test, std_method_test)
{
    vector<Vec3f> colors{ Vec3f(20, 20, 20), Vec3f{ 50, 50, 50 } };
    cv::Scalar mean, std;
    cv::meanStdDev(colors, mean, std);

    EXPECT_EQ(cv::Scalar(35, 35, 35), mean);
    // This computes population standard deviation, so we divide by N, not N - 1.
    float expected_std = sqrtf((powf(35 - 20, 2) + powf(35 - 50, 2)) / 2);
    EXPECT_EQ(cv::Scalar(expected_std, expected_std, expected_std), std);
}

TEST(utility_test, create_rotated_test)
{
    Mat src = Mat::ones(100, 100, CV_32FC1);
    src(Rect(40, 40, 10, 10)) = 0.5;
    auto rotated_srcs = createRotatedImages(src, 0, 10, 10);
    ASSERT_EQ(2, rotated_srcs.size());
    for (int i = 0; i < rotated_srcs.size(); i++) {
        std::string filename = "rotated" + std::to_string(i) + ".exr";
        cv::imwrite(filename, rotated_srcs[i]);
    }
}