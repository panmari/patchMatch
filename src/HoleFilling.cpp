#include "HoleFilling.h"
#include "RandomizedPatchMatch.h"
#include "VotedReconstruction.h"

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

    while (sum(_hole_pyr[_nr_scales])[0] == 0) {
        _nr_scales--;
    }

    _target_area_pyr = std::vector<Mat>(_nr_scales + 1);

    // Set the mean color of the whole image as initial guess.
    Mat mask = _hole_pyr[_nr_scales];
    Mat initial_guess = _img_pyr[_nr_scales].clone();
    Rect low_res_target_rect = computeTargetRect(_img_pyr[_nr_scales], mask, patch_size);
    Scalar mean_color = sum(_img_pyr[_nr_scales]) / (_img_pyr[_nr_scales].cols * _img_pyr[_nr_scales].rows);
    initial_guess.setTo(mean_color, mask);
    initial_guess(low_res_target_rect).copyTo(_target_area_pyr[_nr_scales]);

}

Mat HoleFilling::run() {
    // Set the source to full black in hole.
    Mat source = _img_pyr[_nr_scales];
    source.setTo(Scalar(0, 0 ,0), _hole_pyr[_nr_scales]);
    RandomizedPatchMatch rmp(source, _target_area_pyr[_nr_scales], _patch_size);
    Mat offset_map = rmp.match();
    VotedReconstruction vr(offset_map, source, _patch_size);
    Mat reconstructed = vr.reconstruct();
    return reconstructed;
}

Rect HoleFilling::computeTargetRect(Mat &img, Mat &hole, int patch_size) const {
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
