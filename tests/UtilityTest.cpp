#include "opencv2/ts.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../src/util.h"

using cv::Vec3f;
using pmutil::naiveMeanShift;
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
}