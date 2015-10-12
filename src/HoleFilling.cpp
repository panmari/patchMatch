#include "HoleFilling.h"
#include "RandomizedPatchMatch.h"
#include "VotedReconstruction.h"
#include "util.h"
#include "boost/format.hpp"

using boost::format;
using cv::findNonZero;
using cv::Mat;
using cv::Point;
using cv::pyrUp;
using cv::Rect;
using cv::Scalar;
using std::vector;
using std::max_element;
using std::min_element;

/**
 * Number of iterations for expectation maximization, in our case reconstruction and building of NNF.
 */
const int EM_STEPS = 10;
const bool WEXLER_UPSCALE = false;
const bool DUMP_INTERMEDIARY_RESULTS = false;

namespace {
    static bool compare_by_x(Point a, Point b) {
        return a.x < b.x;
    }

    static bool compare_by_y(Point a, Point b) {
        return a.y < b.y;
    }
}

HoleFilling::HoleFilling(Mat &img, Mat &hole, int patch_size) :
        _img(img), _hole(hole), _patch_size(patch_size),
        _nr_scales((int) log2(std::min(img.cols, img.rows) / (2.f * patch_size))) {
    buildPyramid(img, _img_pyr, _nr_scales);
    buildPyramid(hole, _hole_pyr, _nr_scales);

    // Skip scales where the hole vanishes, i. e. makes up 0 pixels.
    while (sum(_hole_pyr[_nr_scales])[0] == 0) {
        _nr_scales--;
    }

    // Initialize target rects.
    _target_area_pyr = std::vector<Mat>(_nr_scales + 1);
    for (int i = 0; i < _nr_scales + 1; i ++) {
        _target_rect_pyr.push_back(computeTargetRect(_img_pyr[i], _hole_pyr[i], patch_size));
    }
}

Mat HoleFilling::run() {
    // Set the source to full black in hole.
    for (int scale = _nr_scales; scale >= 0; scale--) {
        Mat source = _img_pyr[scale];
        if (scale == _nr_scales) {
            // Make some initial guess, here mean color of whole image.
            // TODO: Do some interpolation of borders for better initial guess.
            Rect low_res_target_rect = _target_rect_pyr[_nr_scales];
            Mat initial_guess = source(low_res_target_rect).clone();
            Scalar mean_color = sum(source) / (source.cols * source.rows);
            initial_guess.setTo(mean_color, _hole_pyr[_nr_scales](low_res_target_rect));
            initial_guess.copyTo(_target_area_pyr[_nr_scales]);

        } else {
            Mat previous_solution = solutionFor(scale + 1);
            Mat upscaled_solution = upscaleSolution(previous_solution);
            // Copy upscaled solution to current target area.
            upscaled_solution(_target_rect_pyr[scale]).copyTo(_target_area_pyr[scale]);
        }
        // Set 'hole' in source, so we will not get trivial solution (i. e. hole is filled with hole).
        // TODO: Possibly set some other value here.
        source.setTo(Scalar(10000, 10000, 10000), _hole_pyr[scale]);
        for (int i = 0; i < EM_STEPS; i++) {
            if (DUMP_INTERMEDIARY_RESULTS) {
                Mat current_solution = solutionFor(scale);
                pmutil::imwrite_lab(str(format("gitter_hole_filled_scale_%d_iter_%02d.exr") % scale % i),
                                    current_solution);
            }
            RandomizedPatchMatch rmp(source, _target_area_pyr[scale], _patch_size);
            Mat offset_map = rmp.match();
            VotedReconstruction vr(offset_map, source, _patch_size);
            Mat reconstructed = vr.reconstruct();
            // Set reconstruction as new 'guess', i. e. set target area to current reconstruction.
            Mat write_back_mask = _hole_pyr[scale](_target_rect_pyr[scale]);
            reconstructed.copyTo(_target_area_pyr[scale], write_back_mask);
        }
    }
    return solutionFor(0);
}

Mat HoleFilling::upscaleSolution(Mat &previous_solution) const {
    Mat upscaled_solution;
    if (WEXLER_UPSCALE) {
        // TODO: There is actually a better method for upscaling, see Wexler2007 Section 3.2
    } else {
        pyrUp(previous_solution, upscaled_solution);
    }
    // pmutil::imwrite_lab(str(format("gitter_hole_upscaled_scale_%d.exr") % scale), upscaled_solution);
    return upscaled_solution;
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