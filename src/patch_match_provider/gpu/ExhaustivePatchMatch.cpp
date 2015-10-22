#include "ExhaustivePatchMatch.h"
#include <boost/progress.hpp>
#include "boost/iostreams/stream.hpp"

using cv::cuda::createTemplateMatching;
using cv::cuda::GpuMat;
using cv::Mat;
using cv::Point;
using cv::Rect;
using cv::Scalar;
using cv::Vec3f;

ExhaustivePatchMatch::ExhaustivePatchMatch(Mat &img, Mat &img2, int patch_size, bool show_progress_bar) :
        _patch_size(patch_size), _show_progress_bar(show_progress_bar) {
    _img.upload(img);
    _img2.upload(img2);
    _cuda_matcher = createTemplateMatching(CV_32FC3, CV_TM_SQDIFF);
    _temp.create(_img.rows - _patch_size + 1, _img.cols - _patch_size + 1, CV_32FC1);
}

void ExhaustivePatchMatch::match(OffsetMap *offset_map_entry) {
    Mat offset_map;
	offset_map.create(_img.rows - _patch_size, _img.cols - _patch_size, CV_32FC3);

	const unsigned long matched_pixels = static_cast<unsigned long>(offset_map.cols * offset_map.rows);

    boost::iostreams::stream<boost::iostreams::null_sink> nullout { boost::iostreams::null_sink{} };
    std::ostream& out = _show_progress_bar ? std::cout : nullout;
    boost::progress_display show_progress(matched_pixels, out);

	for (int x = 0; x < offset_map.cols; x++) {
		for (int y = 0; y < offset_map.rows; y++) {
			Rect rect(x, y, _patch_size, _patch_size);
            GpuMat patch = _img2(rect);
            double minVal; Point minLoc;
            matchSinglePatch(patch, &minVal, &minLoc);
			offset_map.at<Vec3f>(y, x) = Vec3f(minLoc.x - x, minLoc.y - y, static_cast<float>(minVal));
        }
        show_progress += offset_map.rows;
    }
}

void ExhaustivePatchMatch::matchSinglePatch(GpuMat &patch, double *minVal, Point *minLoc) {
    // Do the Matching
    _cuda_matcher->match(_img, patch, _temp);
    // Localizing the best match with minMaxLoc
    cv::cuda::minMaxLoc(_temp, minVal, nullptr, minLoc, nullptr);
    return;
}
