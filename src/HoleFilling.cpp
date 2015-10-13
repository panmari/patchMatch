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
const bool WEXLER_UPSCALE = true;
const bool DUMP_INTERMEDIARY_RESULTS = true;

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
    _target_area_pyr.resize(_nr_scales + 1);
    _target_rect_pyr.resize(_nr_scales + 1);
    _offset_map_pyr.resize(_nr_scales + 1);
    for (int i = 0; i < _nr_scales + 1; i ++) {
        _target_rect_pyr[i] = computeTargetRect(_img_pyr[i], _hole_pyr[i], patch_size);
    }
}

Mat HoleFilling::run() {
    // Set the source to full black in hole.
    for (int scale = _nr_scales; scale >= 0; scale--) {
        Mat source = _img_pyr[scale].clone();
        if (scale == _nr_scales) {
            // Make some initial guess, here mean color of whole image.
            // TODO: Do some interpolation of borders for better initial guess.
            Rect low_res_target_rect = _target_rect_pyr[_nr_scales];
            Mat initial_guess = source(low_res_target_rect).clone();
            Scalar mean_color = sum(source) / (source.cols * source.rows);
            initial_guess.setTo(mean_color, _hole_pyr[_nr_scales](low_res_target_rect));
            initial_guess.copyTo(_target_area_pyr[_nr_scales]);

        } else {
            Mat upscaled_solution = upscaleSolution(scale);
            // Copy upscaled solution to current target area.
            upscaled_solution.copyTo(_target_area_pyr[scale]);
        }
        // Set 'hole' in source, so we will not get trivial solution (i. e. hole is filled with hole).
        // TODO: Possibly set some other value here.
        source.setTo(Scalar(10000, 10000, 10000), _hole_pyr[scale]);
        for (int i = 0; i < EM_STEPS; i++) {
            RandomizedPatchMatch rmp(source, _target_area_pyr[scale], _patch_size);
            _offset_map_pyr[scale] = rmp.match();
            if (DUMP_INTERMEDIARY_RESULTS) {
                cv::String modifier = str(format("scale_%d_iter_%02d") % scale % i);
                Mat current_solution = solutionFor(scale);
                pmutil::imwrite_lab("hole_filled_" + modifier + ".exr", current_solution);
                // Dump nearest patches for every pixel in offset map
                // Mat img_bgr;
                // cvtColor(source, img_bgr, CV_Lab2BGR);
                //pmutil::dumpNearestPatches(_offset_map_pyr[scale], img_bgr, _patch_size, modifier);
            }
            VotedReconstruction vr(_offset_map_pyr[scale], source, _patch_size);
            Mat reconstructed = vr.reconstruct();
            // Set reconstruction as new 'guess', i. e. set target area to current reconstruction.
            Mat write_back_mask = _hole_pyr[scale](_target_rect_pyr[scale]);
            reconstructed.copyTo(_target_area_pyr[scale], write_back_mask);
        }
    }
    return solutionFor(0);
}

Mat HoleFilling::upscaleSolution(int current_scale) const {
    Mat upscaled_target_area;
    if (WEXLER_UPSCALE) {
        // Better method for upscaling, see Wexler2007 Section 3.2
        int previous_scale = current_scale + 1;
        Mat previous_offset_map = _offset_map_pyr[previous_scale];
        Mat source = _img_pyr[current_scale];
        cv::Size upscaled_solution_size((previous_offset_map.cols - 1) * 2 + _patch_size*2,
                                        (previous_offset_map.rows - 1) * 2 + _patch_size*2);
        Rect target_area_rect = _target_rect_pyr[current_scale];
        upscaled_target_area = Mat::zeros(upscaled_solution_size, CV_32FC3);
        Mat count = Mat::zeros(upscaled_solution_size, CV_32FC1);
        for (int x = 0; x < previous_offset_map.cols; x++) {
            for (int y = 0; y < previous_offset_map.rows; y++) {
                // Go over all patches that contain this image.
                cv::Vec3f offset_map_entry = previous_offset_map.at<cv::Vec3f>(y, x);
                int match_x = (x + offset_map_entry[0]) * 2;
                int match_y = (y + offset_map_entry[1]) * 2;
                // Get image data of matching patch
                Rect matching_patch_rect(match_x, match_y, _patch_size * 2, _patch_size * 2);
                Mat matching_patch = source(matching_patch_rect);

                float weight = 1;
                // Add to all pixels at once.
                Rect current_patch_rect(x * 2, y * 2, _patch_size * 2, _patch_size * 2);

                upscaled_target_area(current_patch_rect) += matching_patch * weight;

                // Remember for every pixel, how many patches were added up for later division.
                count(current_patch_rect) += weight;
            }
        }
        vector<Mat> channels(3);
        split(upscaled_target_area, channels);
        for (Mat chan: channels) {
            divide(chan, count, chan);
        }
        merge(channels, upscaled_target_area);
        pmutil::imwrite_lab("wexler_upscaled" + std::to_string(current_scale) + "_full.exr", upscaled_target_area);
        // TODO: Why -1?
        cv::Point2f center(upscaled_solution_size.width / 2.f, upscaled_solution_size.height / 2.f - 1);
        getRectSubPix(upscaled_target_area, target_area_rect.size(), center, upscaled_target_area);
        pmutil::imwrite_lab("wexler_upscaled" + std::to_string(current_scale) + ".exr", upscaled_target_area);
    } else {
        Mat previous_solution = solutionFor(current_scale + 1);
        Mat upscaled_solution;
        pyrUp(previous_solution, upscaled_solution);
        upscaled_target_area = upscaled_solution(_target_rect_pyr[current_scale]);
    }
    return upscaled_target_area;
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

Mat HoleFilling::solutionFor(int scale) const {
    Mat source = _img_pyr[scale].clone();
    Mat write_back_mask = _hole_pyr[scale](_target_rect_pyr[scale]);
    _target_area_pyr[scale].copyTo(source(_target_rect_pyr[scale]), write_back_mask);
    return source;
}