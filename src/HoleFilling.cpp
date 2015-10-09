#include "HoleFilling.h"

using cv::findNonZero;
using cv::Mat;
using cv::Point;
using cv::Rect;
using cv::Scalar;
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

HoleFilling::HoleFilling(Mat &img, Mat &hole, int patch_size) : _img(img), _hole(hole), _patch_size(patch_size),
        _nr_scales((int) log2(std::min(img.cols, img.rows) / (2.f * patch_size))) {
    buildPyramid(img, _img_pyr, _nr_scales);
    buildPyramid(hole, _hole_pyr, _nr_scales);

    _target_rect = computeTargetRect(img, hole, patch_size);

    Rect low_res_target_rect = computeTargetRect(_img_pyr[_nr_scales], _hole_pyr[_nr_scales], patch_size);
    Mat initial_guess = _img_pyr[_nr_scales].clone();
    Mat mask, not_mask;
    mask = _hole_pyr[_nr_scales];
    bitwise_not(mask, not_mask);
    initial_guess.setTo(Scalar(1, 0, 0), _hole_pyr[_nr_scales]);
    // Only preserve colors for mask pixels
    // Fill with average color
    // TODO: Do something better.

    initial_guess(low_res_target_rect).copyTo(_target_area);

//    buildPyramid(img, _img_pyr, _nr_scales);
//    _offset_map_pyr.resize(_nr_scales + 1);

    // TODO: Construct NNF for img, fill hole iteratively by reconstructing the pixels within using matches from the NNF.
    // Questions: Is only used for pixels inside hole? May whole pixels participate in NNF?
    // - Probably yes, but when reconstructing, the contribution of these pixels is weighted down by
    // alpha_b = 1.3 ^ -dist (distance to edge).
    //
}

Rect HoleFilling::computeTargetRect(Mat &img, Mat &hole, int patch_size) {
    vector<Point> non_zero_locations;
    findNonZero(hole, non_zero_locations);
    int min_x = min_element(non_zero_locations.begin(), non_zero_locations.end(), compare_by_x)->x;
    int max_x = max_element(non_zero_locations.begin(), non_zero_locations.end(), compare_by_x)->x;
    int min_y = min_element(non_zero_locations.begin(), non_zero_locations.end(), compare_by_y)->y;
    int max_y = max_element(non_zero_locations.begin(), non_zero_locations.end(), compare_by_y)->y;


    Rect target_rect(Point(min_x - patch_size + 1, min_y - patch_size + 1),
                     Point(max_x + patch_size, max_y + patch_size));

    // Crop to image size.
    target_rect = target_rect & Rect(Point(0, 0), img.size());
    return target_rect;
}
