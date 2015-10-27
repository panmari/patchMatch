#include "HoleFilling.h"
#include "patch_match_provider/RandomizedPatchMatch.h"
#include "VotedReconstruction.h"
#include "util.h"

using boost::format;
using cv::findNonZero;
using cv::countNonZero;
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
const int EM_STEPS = 20;
const bool WEXLER_UPSCALE = true;
const bool DUMP_INTERMEDIARY_RESULTS = true;
const bool DUMP_UPSCALING_DEBUG_OUTPUT = false;

namespace {
    bool compare_by_x(Point a, Point b) {
        return a.x < b.x;
    }

    bool compare_by_y(Point a, Point b) {
        return a.y < b.y;
    }

    int computeNrScales(const Mat &img, int patch_size) {
        int min_dimension = std::min(img.cols, img.rows);
        float min_downscaled_size = 2 * patch_size;
        return static_cast<int>(log2f( min_dimension / min_downscaled_size) + 0.5f);
    }
}

HoleFilling::HoleFilling(Mat &img, Mat &hole, int patch_size) : _patch_size(patch_size),
        _nr_scales(computeNrScales(img, patch_size)) {
    buildPyramid(img, _img_pyr, _nr_scales);
    buildPyramid(hole, _hole_pyr, _nr_scales);

    // Skip scales where the hole vanishes, i. e. makes up 0 pixels.
    while (sum(_hole_pyr[_nr_scales])[0] == 0) {
        _nr_scales--;
    }

    // Initialize target rects.
    _target_area_pyr.resize(_nr_scales + 1);
    _offset_map_pyr.resize(_nr_scales + 1);
    for (int i = 0; i < _nr_scales + 1; i ++) {
        _target_rect_pyr.push_back(computeTargetRect(_img_pyr[i], _hole_pyr[i], patch_size));
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
            Mat low_res_inverted_mask = 255 - _hole_pyr[_nr_scales](low_res_target_rect);
            Mat without_hole;
            source(low_res_target_rect).copyTo(without_hole, low_res_inverted_mask);
            Scalar mean_color = sum(without_hole) / countNonZero(low_res_inverted_mask);
            Mat initial_guess = source(low_res_target_rect).clone();
            initial_guess.setTo(mean_color, _hole_pyr[_nr_scales](low_res_target_rect));
            _target_area_pyr[_nr_scales] = initial_guess;
        } else {
            Mat upscaled_solution = upscaleSolution(scale);
            // Copy upscaled solution at hole region to current target area.
            _target_area_pyr[scale] = source(_target_rect_pyr[scale]).clone();
            // Only copy upscaled solution in hole region.
            Mat hole_mask = _hole_pyr[scale];
            upscaled_solution.copyTo(_target_area_pyr[scale], hole_mask(_target_rect_pyr[scale]));
        }
        // Set 'hole' in source, so we will not get trivial solution (i. e. hole is filled with hole).
        // TODO: Possibly set some other value here.
        source.setTo(Scalar(10000, 10000, 10000), _hole_pyr[scale]);
        RandomizedPatchMatch rmp(source, _target_area_pyr[scale], _patch_size, 0);
        for (int i = 0; i < EM_STEPS; i++) {
            if (DUMP_INTERMEDIARY_RESULTS) {
                double pd = 0;
                if (i > 0) {
                    pd = _offset_map_pyr[scale]->summedDistance();
                }
                const int scale_for_output = _nr_scales - scale;
                cv::String modifier = str(format("scale_%d_iter_%02d_pd_%f") % scale_for_output % i % pd);
                Mat current_solution = solutionFor(scale);
                pmutil::imwrite_lab("hole_filled_" + modifier + ".exr", current_solution);
                // Dump nearest patches for every pixel in offset map
                // Mat img_bgr;
                // cvtColor(source, img_bgr, CV_Lab2BGR);
                //pmutil::dumpNearestPatches(_offset_map_pyr[scale], img_bgr, _patch_size, modifier);
            }
            // TODO: do cleanup again.
            /*
            if (i > 0) {
                // Delete previous offset map.
                delete (_offset_map_pyr[scale]);
            } */
            _offset_map_pyr[scale] = rmp.match();
            VotedReconstruction vr(_offset_map_pyr[scale], source,
                                   rmp.getSourceGradientX(), rmp.getSourceGradientY(),
                                   _patch_size);
            float mean_shift_bandwith_scale = 3 - i * (3 - 0.2f) / (EM_STEPS - 1);
            Mat reconstructed;
            vr.reconstruct(reconstructed, mean_shift_bandwith_scale);
            // Set reconstruction as new 'guess', i. e. set target area to current reconstruction.
            Mat write_back_mask = _hole_pyr[scale](_target_rect_pyr[scale]);
            reconstructed.copyTo(_target_area_pyr[scale], write_back_mask);
            rmp.setTargetArea(_target_area_pyr[scale]);
        }
    }
    for (OffsetMap *offset_map :_offset_map_pyr) {
        delete(offset_map);
    }
    return solutionFor(0);
}

Mat HoleFilling::upscaleSolution(int current_scale) const {
    Mat upscaled_target_area;
    if (WEXLER_UPSCALE) {
        // Better method for upscaling, see Wexler2007 Section 3.2
        int previous_scale = current_scale + 1;
        OffsetMap *previous_offset_map = _offset_map_pyr[previous_scale];
        Mat source = _img_pyr[current_scale];
        Mat gx, gy;
        pmutil::computeGradientX(source, gx);
        pmutil::computeGradientY(source, gy);
        VotedReconstruction vr(previous_offset_map, source, gx, gy, _patch_size, 2);
        // TODO: Find out what mean shift scale works best here.
        vr.reconstruct(upscaled_target_area, 3);

        // Cut out the needed portion of the upscaled target area by
        // projecting top left of previous target area to new scale, compute offset to needed top left.
        Rect prev_target_area_rect = _target_rect_pyr[previous_scale];
        Rect target_area_rect = _target_rect_pyr[current_scale];
        Point offset = target_area_rect.tl() - prev_target_area_rect.tl() * 2;
        Rect cutout_rect(offset, target_area_rect.size());
        // In case our target area was right at the edge of the image, we might need to increase the size a bit here.
        if (cutout_rect.x + cutout_rect.width > upscaled_target_area.cols ||
                cutout_rect.y + cutout_rect.height > upscaled_target_area.rows) {
            copyMakeBorder(upscaled_target_area, upscaled_target_area, 0, 2, 0, 2, cv::BORDER_REFLECT);
        }

        upscaled_target_area = upscaled_target_area(cutout_rect);

        if (DUMP_UPSCALING_DEBUG_OUTPUT) {
            Rect upscaled_rect(prev_target_area_rect.x * 2, prev_target_area_rect.y * 2,
                               prev_target_area_rect.width * 2, prev_target_area_rect.height * 2);
            cv::rectangle(source, upscaled_rect, Scalar(100, 10, 10));
            cv::rectangle(source, target_area_rect, Scalar(0, 0, 0));
            source(Rect(Point(0, 0), _hole_pyr[current_scale].size())).setTo(cv::Scalar(0, 0, 0),
                                                                             _hole_pyr[current_scale]);

            pmutil::imwrite_lab("wexler_upscaled" + std::to_string(current_scale) + "_full.exr", upscaled_target_area);
            pmutil::imwrite_lab("wexler_upscaled" + std::to_string(current_scale) + ".exr", upscaled_target_area);
            pmutil::imwrite_lab("wexler_upscaled" + std::to_string(current_scale) + "_target_area_in_img.exr", source);
        }
    } else {
        Mat previous_solution = solutionFor(current_scale + 1);
        Mat upscaled_solution;
        pyrUp(previous_solution, upscaled_solution);
        upscaled_target_area = upscaled_solution(_target_rect_pyr[current_scale]);
    }
    return upscaled_target_area;
}

Rect HoleFilling::computeTargetRect(const Mat &img, const Mat &hole, int patch_size) const {
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
