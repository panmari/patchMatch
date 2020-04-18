#include "gtest/gtest.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#ifdef OpenCV_CUDA_VERSION
#include "../src/patch_match_provider/gpu/ExhaustivePatchMatch.h"
#else
#include "../src/patch_match_provider/cpu/ExhaustivePatchMatch.h"
#endif
#include "../src/patch_match_provider/RandomizedPatchMatch.h"
#include "../src/util.h"

using cv::imread;
using cv::Mat;
using cv::Rect;
using cv::Scalar;
using cv::Vec3f;
using pmutil::convert_for_computation;
using std::shared_ptr;

const double EPSILON = 1e-3;

TEST(randomized_patch_match_test, on_two_identical_trivial_images)
{
    Mat img1 = Mat::ones(20, 20, CV_32FC1);
    Mat img2 = Mat::ones(20, 20, CV_32FC1);

    RandomizedPatchMatch rpm(img1, img2.size(), 7, 0);
    rpm.setTargetArea(img2);
    shared_ptr<OffsetMap> diff = rpm.match();
    double overall_ssd = diff->summedDistance();

    ASSERT_NEAR(0.0, overall_ssd, EPSILON);
}

TEST(randomized_patch_match_test, on_two_very_different_trivial_images)
{
    Mat img1 = Mat::zeros(20, 20, CV_32FC1);
    Mat img2 = Mat::ones(20, 20, CV_32FC1);

    int patch_size = 7;
    RandomizedPatchMatch rpm(img1, img2.size(), patch_size, 0.f);
    rpm.setTargetArea(img2);
    shared_ptr<OffsetMap> diff = rpm.match();
    double overall_ssd = diff->summedDistance();

    // We expect for every patch (size - patch_size)^2 the maximum deviation of 7*7 (every pixel has SSD 1)
    int error_per_patch = patch_size * patch_size;
    double expected_ssd = (20 + 1 - patch_size) * (20 + 1 - patch_size) * error_per_patch;
    ASSERT_NEAR(expected_ssd, overall_ssd, EPSILON);
}

TEST(randomized_patch_match_test, all_offsets_inside_image_on_random_images)
{
    Mat img1 = Mat::zeros(20, 20, CV_32FC1);
    randu(img1, Scalar::all(0.0), Scalar::all(1.0f));
    Mat img2 = Mat::ones(40, 40, CV_32FC1);
    randu(img2, Scalar::all(0.0), Scalar::all(1.0f));
    int patch_size = 7;

    RandomizedPatchMatch rpm(img1, img2.size(), patch_size, 0.f);
    rpm.setTargetArea(img2);
    shared_ptr<OffsetMap> diff = rpm.match();
    for (int x = 0; x < diff->_width; x++) {
        for (int y = 0; y < diff->_height; y++) {
            OffsetMapEntry d = diff->at(y, x);
            int matching_patch_x = x + d.offset.x;
            int matching_patch_y = y + d.offset.y;
            ASSERT_GE(matching_patch_x, 0) << "Failed for offset in (" << x << "," << y << ")";
            ASSERT_GE(matching_patch_y, 0) << "Failed for offset in (" << x << "," << y << ")";
            ASSERT_LT(matching_patch_x, img2.cols) << "Failed for offset in (" << x << "," << y << ")";
            ASSERT_LT(matching_patch_y, img2.rows) << "Failed for offset in (" << x << "," << y << ")";
        }
    }
}

TEST(randomized_patch_match_test, images_with_displaced_rectangles_should_produce_little_dist)
{
    Mat img1 = Mat::ones(50, 50, CV_32FC1);
    Mat img2 = Mat::ones(50, 50, CV_32FC1);

    // Black rectangle on image 1
    img1(Rect(10, 10, 5, 5)) = 0;
    // Black rectangle on image 2 in slightly different location.
    img2(Rect(20, 25, 5, 5)) = 0;
    int patch_size = 7;

    RandomizedPatchMatch rpm(img1, img2.size(), patch_size, 0.f);
    rpm.setTargetArea(img2);
    shared_ptr<OffsetMap> diff = rpm.match();
    imwrite("offset_map_visualized.exr", diff->toColorCodedImage());
    imwrite("dist_img.exr", diff->getDistanceImage());

    double overall_ssd = diff->summedDistance();
    double expected_ssd = 0;
    ASSERT_NEAR(expected_ssd, overall_ssd, EPSILON);
}


TEST(randomized_patch_match_test, should_be_close_to_exhaustive_patch_match)
{
    // TODO: this is not very true anymore, since exhaustive search does not support offsets.
	Mat source = imread("test_images/sonne1.PNG");
	Mat target = imread("test_images/sonne2.PNG");
	const float resize_factor = 0.25f;
	pmutil::convert_for_computation(source, resize_factor);
	pmutil::convert_for_computation(target, resize_factor);
	const int patch_size = 7;
    RandomizedPatchMatch rpm(source, target.size(), patch_size, 0.f);
    rpm.setTargetArea(target);
    shared_ptr<OffsetMap> diff_rpm = rpm.match();

	ExhaustivePatchMatch epm(source, target, patch_size);
    shared_ptr<OffsetMap> diff_epm = epm.match();

	const int nr_pixels = target.cols * target.rows;
	double mean_ssd_rpm = diff_rpm->summedDistance() / nr_pixels;

	double mean_ssd_epm = diff_epm->summedDistance() / nr_pixels;

	// This is in L*a*b* space, so the errors are quite high.
    // Still, rpm should have lower error since rotations are possible there.
	ASSERT_LT(mean_ssd_rpm, mean_ssd_epm);
}