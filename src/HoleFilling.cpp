#include "HoleFilling.h"
#include "RandomizedPatchMatch.h"
#include "VotedReconstruction.h"
#include "opencv2/highgui/highgui.hpp"
#include "util.h"

using cv::findNonZero;
using cv::Mat;
using cv::Point;
using cv::Rect;
using cv::Scalar;
using std::vector;
using std::max_element;
using std::min_element;

/**
 * Number of iterations for expectation maximizationm, in our case reconstruction and building of NNF.
 */
const int EM_STEPS = 20;
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

    // Initialize target rects.
    _target_area_pyr = std::vector<Mat>(_nr_scales + 1);
    for (int i = 0; i < _nr_scales + 1; i ++) {
        _target_rect_pyr.push_back(computeTargetRect(_img_pyr[i], _hole_pyr[i], patch_size));
    }


    // Set the mean color of the whole image as initial guess.
    Mat initial_guess = _img_pyr[_nr_scales].clone();
    Rect low_res_target_rect = _target_rect_pyr[_nr_scales];
    Scalar mean_color = sum(_img_pyr[_nr_scales]) / (_img_pyr[_nr_scales].cols * _img_pyr[_nr_scales].rows);
    initial_guess.setTo(mean_color, _hole_pyr[_nr_scales]);
    initial_guess(low_res_target_rect).copyTo(_target_area_pyr[_nr_scales]);
}

Mat HoleFilling::run() {
    // Set the source to full black in hole.
    Mat source = _img_pyr[_nr_scales];
    source.setTo(Scalar(0, 0, 0), _hole_pyr[_nr_scales]);
    //pmutil::imwrite_lab("hole_filling_source_with_black_hole.exr", source);
    for (int i = 0; i < EM_STEPS; i++) {
        RandomizedPatchMatch rmp(source, _target_area_pyr[_nr_scales], _patch_size);
        Mat offset_map = rmp.match();
        VotedReconstruction vr(offset_map, source, _patch_size);
        Mat reconstructed = vr.reconstruct();
        // Set reconstruction as new 'guess', i. e. set target area to current reconstruction.
        Mat write_back_mask = _hole_pyr[_nr_scales](_target_rect_pyr[_nr_scales]);
        reconstructed.copyTo(_target_area_pyr[_nr_scales], write_back_mask);
        // Dumps current solution
        Mat current = solutionFor(_nr_scales);
        cvtColor(current, current, CV_Lab2BGR);
        imwrite("gitter_hole_filled_scale_" + std::to_string(_nr_scales) + "_iter_" + std::to_string(i) + ".exr", current);
    }
    return solutionFor(_nr_scales);
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

Mat HoleFilling::solutionFor(int scale) {
    Mat source = _img_pyr[scale].clone();
    Mat write_back_mask = _hole_pyr[scale](_target_rect_pyr[scale]);
    _target_area_pyr[scale].copyTo(source(_target_rect_pyr[scale]), write_back_mask);
    return source;
}