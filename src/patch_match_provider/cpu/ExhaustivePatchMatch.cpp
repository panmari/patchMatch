#include "ExhaustivePatchMatch.h"
#include <boost/progress.hpp>
#include "boost/iostreams/stream.hpp"

using cv::Mat;
using cv::matchTemplate;
using cv::minMaxLoc;
using cv::Point;
using cv::Rect;
using cv::Scalar;
using cv::Vec3f;

ExhaustivePatchMatch::ExhaustivePatchMatch(const Mat &source, const Mat &target, int patch_size,
                                           bool show_progress_bar) : _source(source), _target(target),
        _patch_size(patch_size), _show_progress_bar(show_progress_bar) {
    _temp.create(source.rows - _patch_size + 1, source.cols - _patch_size + 1, CV_32FC1);
}

OffsetMap* ExhaustivePatchMatch::match() {
    Mat offset_map;
	offset_map.create(_target.rows - _patch_size, _target.cols - _patch_size, CV_32FC3);

	const unsigned long matched_pixels = static_cast<unsigned long>(offset_map.cols * offset_map.rows);

    boost::iostreams::stream<boost::iostreams::null_sink> nullout { boost::iostreams::null_sink{} };
    std::ostream& out = _show_progress_bar ? std::cout : nullout;
    boost::progress_display show_progress(matched_pixels, out);

	for (int x = 0; x < offset_map.cols; x++) {
		for (int y = 0; y < offset_map.rows; y++) {
			Rect rect(x, y, _patch_size, _patch_size);
            Mat patch = _target(rect);
            double minVal; Point minLoc;
            matchSinglePatch(patch, &minVal, &minLoc);
			offset_map.at<Vec3f>(y, x) = Vec3f(minLoc.x - x, minLoc.y - y, static_cast<float>(minVal));
        }
        show_progress += offset_map.rows;
    }
    //TODO: fix this.
    OffsetMap* om = new OffsetMap(0,0);
    return om;
}

void ExhaustivePatchMatch::matchSinglePatch(const Mat &patch, double *minVal, Point *minLoc) const {
    // Do the Matching
    matchTemplate(_source, patch, _temp, cv::TM_SQDIFF);
    // Localizing the best match with minMaxLoc
    minMaxLoc(_temp, minVal, nullptr, minLoc, nullptr);
}
