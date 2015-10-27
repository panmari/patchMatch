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
using std::shared_ptr;

ExhaustivePatchMatch::ExhaustivePatchMatch(const Mat &source, const Mat &target, int patch_size,
                                           bool show_progress_bar) : _source(source), _target(target),
        _patch_size(patch_size), _show_progress_bar(show_progress_bar) {
    _temp.create(source.rows - _patch_size + 1, source.cols - _patch_size + 1, CV_32FC1);
}

shared_ptr<OffsetMap> ExhaustivePatchMatch::match() {
    auto offset_map = shared_ptr<OffsetMap>(new OffsetMap(_target.cols - _patch_size + 1,
                                                          _target.rows - _patch_size + 1));

    const unsigned long matched_pixels = static_cast<unsigned long>(offset_map->_width * offset_map->_height);

    boost::iostreams::stream<boost::iostreams::null_sink> nullout { boost::iostreams::null_sink{} };
    std::ostream& out = _show_progress_bar ? std::cout : nullout;
    boost::progress_display show_progress(matched_pixels, out);

    for (int x = 0; x < offset_map->_width; x++) {
        for (int y = 0; y < offset_map->_height; y++) {
			Rect rect(x, y, _patch_size, _patch_size);
            Mat patch = _target(rect);
            OffsetMapEntry *entry = offset_map->ptr(y, x);
            double minVal; Point min_loc;
            matchSinglePatch(patch, &minVal, &min_loc);
            entry->distance = static_cast<float>(minVal);
            entry->offset = Point(min_loc.x - x, min_loc.y - y);
        }
        show_progress += offset_map->_width;
    }
    return offset_map;
}

void ExhaustivePatchMatch::matchSinglePatch(const Mat &patch, double *minVal, Point *minLoc) const {
    // Do the Matching
    matchTemplate(_source, patch, _temp, cv::TM_SQDIFF);
    // Localizing the best match with minMaxLoc
    minMaxLoc(_temp, minVal, nullptr, minLoc, nullptr);
}
