#include "HoleFilling.h"

using cv::findNonZero;
using cv::Mat;
using cv::Point;
using cv::Rect;
using std::vector;
using std::max_element;
using std::min_element;

namespace {
    static bool compare_by_x(Point a, Point b) {
        return a.x < b.x;
    }

    static bool compare_by_y(Point a, Point b) {
        return a.y < b.y;
    }
}

HoleFilling::HoleFilling(Mat &img, Mat &hole, int patch_size) : _img(img), _hole(hole), _patch_size(patch_size) {
    vector<Point> non_zero_locations;
    findNonZero(_hole, non_zero_locations);
    int min_x = (*min_element(non_zero_locations.begin(), non_zero_locations.end(), compare_by_x)).x;
    int max_x = (*max_element(non_zero_locations.begin(), non_zero_locations.end(), compare_by_x)).x;
    int min_y = (*min_element(non_zero_locations.begin(), non_zero_locations.end(), compare_by_y)).y;
    int max_y = (*max_element(non_zero_locations.begin(), non_zero_locations.end(), compare_by_y)).y;

    _target_rect = Rect(Point(min_x - _patch_size + 1, min_y - _patch_size + 1),
                        Point(max_x + _patch_size, max_y + _patch_size));
    // Crop to image size.
    _target_rect = _target_rect & Rect(Point(0, 0), _img.size());
    _target_area = img(_target_rect);

    // TODO: Construct NNF for img, fill hole iteratively by reconstructing the pixels within using matches from the NNF.
    // Questions: Is only used for pixels inside hole? May whole pixels participate in NNF?
    // - Probably yes, but when reconstructing, the contribution of these pixels is weighted down by
    // alpha_b = 1.3 ^ -dist (distance to edge).
    //
}
