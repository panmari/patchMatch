#include "RandomizedPatchMatch.h"
#include "opencv2/highgui/highgui.hpp"
#include "../util.h"
#include <iostream>

using cv::addWeighted;
using cv::buildPyramid;
using cv::flip;
using cv::Mat;
using cv::Point;
using cv::Scalar;
using cv::Size;
using cv::String;
using cv::Rect;
using cv::RNG;
using cv::Vec3f;
using std::max;
using pmutil::ssd_unsafe;
using pmutil::computeGradientX;
using pmutil::computeGradientY;

/**
 * Number times propagation/random_search is executed for every patch per iteration.
 */
const int ITERATIONS_PER_SCALE = 5;
/**
 * If true, does try to improve patches by doing random search. Else, only propagation is used. Default: true.
 */
const bool RANDOM_SEARCH = true;
const bool MERGE_UPSAMPLED_OFFSETS = true;
const float ALPHA = 0.5; // Used to modify random search radius. Higher alpha means more random searches.

RandomizedPatchMatch::RandomizedPatchMatch(const cv::Mat &source, const cv::Mat &target, int patch_size, float lambda) :
        _patch_size(patch_size), _max_search_radius(max(target.cols, target.rows)),
        _nr_scales(findNumberScales(source, target, patch_size)), _lambda(lambda) {
    buildPyramid(source, _source_pyr, _nr_scales);
    buildPyramid(target, _target_pyr, _nr_scales);
    for (Mat scaled_source: _source_pyr) {
        _source_rect_pyr.push_back(Rect(Point(0,0), scaled_source.size()));
        Mat gx;
        computeGradientX(scaled_source, gx);
        _source_grad_x_pyr.push_back(gx);
        Mat gy;
        computeGradientY(scaled_source, gy);
        _source_grad_y_pyr.push_back(gy);
    }
    for (Mat scaled_target: _target_pyr) {
        Mat lap;
        Mat gx;
        computeGradientX(scaled_target, gx);
        _target_grad_x_pyr.push_back(gx);
        Mat gy;
        computeGradientY(scaled_target, gy);
        _target_grad_y_pyr.push_back(gy);
    }
}

OffsetMap* RandomizedPatchMatch::match() {
    RNG rng( 0xFFFFFFFF );

    OffsetMap *previous_offset_map;
    for (int scale = _nr_scales; scale >= 0; scale--) {
        Mat source = _source_pyr[scale];
        Mat target = _target_pyr[scale];
        const int width = target.cols - _patch_size + 1;
        const int height = target.rows - _patch_size + 1;
        OffsetMap *offset_map = new OffsetMap(width, height);
        initializeWithRandomOffsets(source, target, scale, offset_map);
        bool isFlipped = false;
        for (int i = 0; i < ITERATIONS_PER_SCALE; i++) {
            // After half the iterations, merge the lower resolution offset where they're better.
            // This has to be done in an 'even' iteration because of the flipping.
            if (MERGE_UPSAMPLED_OFFSETS && scale != _nr_scales && i == ITERATIONS_PER_SCALE / 2) {
                for (int x = 0; x < previous_offset_map->_width; x++) {
                    for (int y = 0; y < previous_offset_map->_height; y++) {
                        // Only check one corresponding pixel, will get propagated to adjacent pixels.
                        // TODO: Check 4 corresponding pixels in higher resolution offset image.
                        Point lower_offset = previous_offset_map->at(y, x).offset;
                        Point candidate_offset = lower_offset * 2;
                        Rect candidate_rect((lower_offset.x + x) * 2, (lower_offset.y + y) * 2,
                                            _patch_size, _patch_size);
                        Rect current_patch_rect(x * 2, y * 2, _patch_size, _patch_size);
                        OffsetMapEntry* current_offset = offset_map->ptr(y * 2, x * 2);
                        updateOffsetMapEntryIfBetter(current_patch_rect, candidate_offset,
                                                     candidate_rect, scale, current_offset);
                    }
                }
            }
            for (int x = 0; x < offset_map->_width; x++) {
                for (int y = 0; y < offset_map->_height; y++) {
                    OffsetMapEntry *offset_map_entry = offset_map->ptr(y, x);

                    // If image is flipped, we need to get x and y coordinates unflipped for getting the right offset.
                    int x_unflipped, y_unflipped;
                    if (isFlipped) {
                        x_unflipped = offset_map->_width - 1 - x;
                        y_unflipped = offset_map->_height - 1 - y;
                    } else {
                        x_unflipped = x;
                        y_unflipped = y;
                    }
                    Rect target_patch_rect(x_unflipped, y_unflipped, _patch_size, _patch_size);

                    if (x > 0) {
                        OffsetMapEntry offsetLeft = offset_map->at(y, x - 1);
                        Rect rectLeft(offsetLeft.offset.x + x_unflipped, offsetLeft.offset.y + y_unflipped,
                                      _patch_size, _patch_size);
                        updateOffsetMapEntryIfBetter(target_patch_rect, offsetLeft.offset, rectLeft, scale, offset_map_entry);
                    }
                    if (y > 0) {
                        OffsetMapEntry offsetUp = offset_map->at(y - 1, x);
                        Rect rectUp(offsetUp.offset.x + x_unflipped, offsetUp.offset.y + y_unflipped,
                                    _patch_size, _patch_size);
                        updateOffsetMapEntryIfBetter(target_patch_rect, offsetUp.offset, rectUp, scale, offset_map_entry);
                    }


                    if (RANDOM_SEARCH) {
                        Point current_offset = offset_map_entry->offset;
                        float current_search_radius = _max_search_radius;
                        while (current_search_radius > 1) {
                            Point random_point = Point(rng.uniform(-1.f, 1.f) * current_search_radius,
                                                       rng.uniform(-1.f, 1.f) * current_search_radius);
                            Point random_offset = current_offset + random_point;
                            Rect random_rect(x_unflipped + random_offset.x,
                                             y_unflipped + random_offset.y, _patch_size, _patch_size);

                            updateOffsetMapEntryIfBetter(target_patch_rect, random_offset, random_rect, scale,
                                                         offset_map_entry);

                            current_search_radius *= ALPHA;
                        }
                    }
                }
            }
            // dumpOffsetMapToFile(offset_map, "_scale_" + std::to_string(s) + "_i_" + std::to_string(i));
            // Every second iteration, we go the other way round (start at bottom, propagate from right and down).
            // This effect can be achieved by flipping the matrix after every iteration.
            offset_map->flip();
        }
        if (offset_map->isFlipped()) {
            // Correct orientation if we're still in flipped state.
            offset_map->flip();
        }
        delete previous_offset_map;
        previous_offset_map = offset_map;
    }
    return previous_offset_map;
}

void RandomizedPatchMatch::updateOffsetMapEntryIfBetter(const Rect &target_rect, const Point &candidate_offset,
                                                        const Rect &candidate_rect, const int scale,
                                                        OffsetMapEntry *offset_map_entry) const {
    // Check if it's fully inside, only try to update then
    if ((_source_rect_pyr[scale] & candidate_rect) == candidate_rect) {
        float previous_distance = offset_map_entry->distance;
		float ssd_value = patchDistance(candidate_rect, target_rect, scale, previous_distance);
        if (ssd_value < previous_distance) {
            offset_map_entry->offset = candidate_offset;
            offset_map_entry->distance = ssd_value;
        }
    }

}

void RandomizedPatchMatch::initializeWithRandomOffsets(const Mat &source_img, const Mat &target_img, const int scale,
                                                       OffsetMap *offset_map) const {
    // Seed random generator to have reproducable results.
    // TODO: Use a better initialization to get better results over multiple EM-Steps.
    srand(target_img.rows * target_img.cols);
    for (int x = 0; x < offset_map->_width; x++) {
        for (int y = 0; y < offset_map->_height; y++) {
            // Choose offset carefully, so resulting point (when added to current coordinate), is not outside image.
            int randomX = (rand() % (source_img.cols - _patch_size)) - x;
            int randomY = (rand() % (source_img.rows - _patch_size)) - y;

            Rect current_patch_rect(x, y, _patch_size, _patch_size);
            Rect random_rect = Rect(x + randomX, y + randomY, _patch_size, _patch_size);
			float inital_dist = patchDistance(random_rect, current_patch_rect, scale);
            auto entry = offset_map->ptr(y, x);
            entry->offset = Point(randomX, randomY);
            entry->distance = inital_dist;
        }
    }
}

void RandomizedPatchMatch::dumpOffsetMapToFile(Mat &offset_map, String filename_modifier) const {
    Mat xoffsets, yoffsets, diff, normed;
    Mat out[] = {xoffsets, yoffsets, diff};
    split(offset_map, out);
    Mat angles = Mat::zeros(offset_map.size(), CV_32FC1);
    Mat magnitudes = Mat::zeros(offset_map.size(), CV_32FC1);

    // Produce some nice to look at output by coding angle to best patch as hue, magnitude as saturation.
    for (int x = 0; x < offset_map.cols; x++) {
        for (int y = 0; y < offset_map.rows; y++) {
            Vec3f offset_map_entry = offset_map.at<Vec3f>(y, x);
            float x_offset = offset_map_entry[0];
            float y_offset = offset_map_entry[1];
            float angle = atan2(x_offset, y_offset);
            if (angle < 0)
                angle += CV_2PI;
            angles.at<float>(y, x) = angle / CV_2PI * 360;
            magnitudes.at<float>(y, x) = sqrt(x_offset*x_offset + y_offset*y_offset);
        }
    }
    normalize(magnitudes, magnitudes, 0, 1, cv::NORM_MINMAX, CV_32FC1, Mat() );
    Mat hsv_array[] = {angles, magnitudes, Mat::ones(offset_map.size(), CV_32FC1)};
    Mat hsv;
    cv::merge(hsv_array, 3, hsv);
    cvtColor(hsv, hsv, CV_HSV2BGR);
    imwrite("hsv_offsets" + filename_modifier + ".exr", hsv);

    // Dump unnormalized values for inspection.
    Mat offset_map_bgr;
    cvtColor(offset_map, offset_map_bgr, CV_RGB2BGR);
    imwrite("full_nnf" + filename_modifier + ".exr", offset_map_bgr);
    normalize(out[2], normed, 0, 1, cv::NORM_MINMAX, CV_32FC1, Mat() );
    imwrite("min_dist_img_normalized" + filename_modifier + ".exr", normed);
}

int RandomizedPatchMatch::findNumberScales(const Mat &source, const Mat &target, int patch_size) const {
    int min_dimension = std::min(std::min(std::min(source.cols, source.rows), target.cols), target.rows);
    return (int) log2( min_dimension/ (2.f * patch_size));
}

float RandomizedPatchMatch::patchDistance(const cv::Rect &source_rect, const cv::Rect &target_rect, const int scale,
                                          const float previous_dist) const {
    Mat source_patch = _source_pyr[scale](source_rect);
    Mat target_patch = _target_pyr[scale](target_rect);
    double ssd_img = ssd_unsafe(source_patch, target_patch, previous_dist);

    Mat source_grad_x_patch = _source_grad_x_pyr[scale](source_rect);
    Mat target_grad_x_patch = _target_grad_x_pyr[scale](target_rect);
    double ssd_grad_x = ssd_unsafe(source_grad_x_patch, target_grad_x_patch);

    Mat source_grad_y_patch = _source_grad_y_pyr[scale](source_rect);
    Mat target_grad_y_patch = _target_grad_y_pyr[scale](target_rect);
    double ssd_grad_y = ssd_unsafe(source_grad_y_patch, target_grad_y_patch);

    return static_cast<float>(ssd_img + _lambda * (ssd_grad_x + ssd_grad_y));
}
