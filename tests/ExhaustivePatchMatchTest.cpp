#include "opencv2/ts.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#ifdef OpenCV_CUDA_VERSION
#include "../src/patch_match_provider/gpu/ExhaustivePatchMatch.h"
#else
#include "../src/patch_match_provider/cpu/ExhaustivePatchMatch.h"
#endif

using cv::Mat;
using cv::Scalar;
using cv::Vec3f;

const double EPSILON = 1e-4;

TEST(exhaustive_patch_match_test, on_two_identical_trivial_images)
{
	Mat img1 = Mat::ones(20, 20, CV_32FC1);
	Mat img2 = Mat::ones(20, 20, CV_32FC1);

	ExhaustivePatchMatch epm(img1, img2, 7);
	OffsetMap* diff = epm.match();
	double overall_ssd = diff->summedDistance();

	delete(diff);

	ASSERT_NEAR(0.0, overall_ssd, EPSILON);
}

TEST(exhaustive_patch_match_test, on_two_very_different_trivial_images)
{
	Mat img1 = Mat::zeros(20, 20, CV_32FC1);
	Mat img2 = Mat::ones(20, 20, CV_32FC1);

	ExhaustivePatchMatch epm(img1, img2, 7);
	OffsetMap* diff = epm.match();
	double overall_ssd = diff->summedDistance();
	delete(diff);
	// We expect for every patch (size - patch_size)^2 the maximum deviation of 7*7 (every pixel has SSD 1)
	double expected_ssd = (20 - 7 + 1) * (20 - 7 + 1) * 7 * 7;
	ASSERT_NEAR(expected_ssd, overall_ssd, EPSILON);
}

TEST(exhaustive_patch_match_test, all_offsets_inside_image_on_random_images)
{
	Mat img1 = Mat::zeros(20, 20, CV_32FC1);
	randu(img1, Scalar::all(0.0), Scalar::all(1.0f));
	Mat img2 = Mat::ones(40, 20, CV_32FC1);
	randu(img2, Scalar::all(0.0), Scalar::all(1.0f));

	ExhaustivePatchMatch epm(img1, img2, 7);
	OffsetMap* diff = epm.match();
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
	delete(diff);
}

TEST(exhaustive_patch_match_test, on_two_equal_random_images_offsets_should_be_zero)
{
	Mat img1 = Mat::zeros(20, 20, CV_32FC1);
	randu(img1, 0.f, 1.f);
	Mat img2 = img1.clone();

	ExhaustivePatchMatch epm(img1, img2, 7);
	OffsetMap* diff = epm.match();
	double overall_ssd = diff->summedDistance();
	const double expected_ssd = 0;
	ASSERT_NEAR(expected_ssd, overall_ssd, EPSILON);

	// All offsets should be (0, 0).
	for (int x = 0; x < diff->_width; x++) {
		for (int y = 0; y < diff->_height; y++) {
			OffsetMapEntry d = diff->at(y, x);
			ASSERT_EQ(0, d.offset.x);
			ASSERT_EQ(0, d.offset.y);
		}
	}
	delete(diff);
}