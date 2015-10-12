#include "RandomizedPatchMatch.h"
#include "opencv2/highgui/highgui.hpp"
#include "util.h"
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>

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
using pmutil::ssd;

const int ITERATIONS_PER_SCALE = 5;
const bool RANDOM_SEARCH = true;
const bool MERGE_UPSAMPLED_OFFSETS = true;
const float ALPHA = 0.5; // Used to modify random search radius. Higher alpha means more random searches.

RandomizedPatchMatch::RandomizedPatchMatch(cv::Mat &source, cv::Mat &target, int patch_size) :
        _patch_size(patch_size), _max_search_radius(max(target.cols, target.rows)),
        _nr_scales(findNumberScales(source, target, patch_size)) {
    buildPyramid(source, _source_pyr, _nr_scales);
    buildPyramid(target, _target_pyr, _nr_scales);
    _offset_map_pyr.resize(_nr_scales + 1);
}

cv::Mat RandomizedPatchMatch::match() {
    RNG rng( 0xFFFFFFFF );

    for (int s = _nr_scales; s >= 0; s--) {
        Mat source = _source_pyr[s];
        Mat target = _target_pyr[s];
        Mat offset_map;
        initializeWithRandomOffsets(source, target, offset_map);
        bool isFlipped = false;
        for (int i = 0; i < ITERATIONS_PER_SCALE; i++) {
            // After half the iterations, merge the lower resolution offset where they're better.
            // This has to be done in an 'even' iteration because of the flipping.
            if (MERGE_UPSAMPLED_OFFSETS && s != _nr_scales && i == ITERATIONS_PER_SCALE / 2) {
                Mat lower_offset_map = _offset_map_pyr[s + 1];
                for (int x = 0; x < lower_offset_map.cols; x++) {
                    for (int y = 0; y < lower_offset_map.rows; y++) {
                        // Only check one corresponding pixel, will get propagated to adjacent pixels.
                        // TODO: Check 4 corresponding pixels in higher resolution offset image.
                        Vec3f lower_offset = lower_offset_map.at<Vec3f>(y, x);
                        Point candidate_offset(lower_offset[0] * 2, lower_offset[1] * 2);
                        Rect candidate_rect((lower_offset[0] + x) * 2, (lower_offset[1] + y) * 2,
                                            _patch_size, _patch_size);
                        Rect current_patch_rect(x * 2, y * 2, _patch_size, _patch_size);
                        Mat current_patch = target(current_patch_rect);
                        Vec3f* current_offset = offset_map.ptr<Vec3f>(y * 2, x * 2);
                        updateOffsetMapEntryIfBetter(current_patch, candidate_offset,
                                                     candidate_rect, source, current_offset);
                    }
                }
            }
            for (int x = 0; x < offset_map.cols; x++) {
                for (int y = 0; y < offset_map.rows; y++) {
                    Vec3f *offset_map_entry = offset_map.ptr<Vec3f>(y, x);

                    // If image is flipped, we need to get x and y coordinates unflipped for getting the right offset.
                    int x_unflipped, y_unflipped;
                    if (isFlipped) {
                        x_unflipped = offset_map.cols - 1 - x;
                        y_unflipped = offset_map.rows - 1 - y;
                    } else {
                        x_unflipped = x;
                        y_unflipped = y;
                    }
                    Rect currentPatchRect(x_unflipped, y_unflipped, _patch_size, _patch_size);
                    Mat currentPatch = target(currentPatchRect);

                    if (x > 0) {
                        Vec3f offsetLeft = offset_map.at<Vec3f>(y, x - 1);
                        Rect rectLeft((int) offsetLeft[0] + x_unflipped, (int) offsetLeft[1] + y_unflipped,
                                      _patch_size, _patch_size);
                        Point offsetLeftPoint(offsetLeft[0], offsetLeft[1]);
                        updateOffsetMapEntryIfBetter(currentPatch, offsetLeftPoint, rectLeft, source, offset_map_entry);
                    }
                    if (y > 0) {
                        Vec3f offsetUp = offset_map.at<Vec3f>(y - 1, x);
                        Rect rectUp((int) offsetUp[0] + x_unflipped, (int) offsetUp[1] + y_unflipped,
                                    _patch_size, _patch_size);
                        Point offsetUpPoint(offsetUp[0], offsetUp[1]);
                        updateOffsetMapEntryIfBetter(currentPatch, offsetUpPoint, rectUp, source, offset_map_entry);
                    }

                    Point current_offset((*offset_map_entry)[0], (*offset_map_entry)[1]);

                    if (RANDOM_SEARCH) {
                        float current_search_radius = _max_search_radius;
                        while (current_search_radius > 1) {
                            Point random_point = Point(rng.uniform(-1.f, 1.f) * current_search_radius,
                                                       rng.uniform(-1.f, 1.f) * current_search_radius);
                            Point random_offset = current_offset + random_point;
                            Rect random_rect(x_unflipped + random_offset.x,
                                             y_unflipped + random_offset.y, _patch_size, _patch_size);

                            updateOffsetMapEntryIfBetter(currentPatch, random_offset, random_rect, source,
                                                         offset_map_entry);

                            current_search_radius *= ALPHA;
                        }
                    }
                }
            }
            dumpOffsetMapToFile(offset_map, "_scale_" + std::to_string(s) + "_i_" + std::to_string(i));
            // Every second iteration, we go the other way round (start at bottom, propagate from right and down).
            // This effect can be achieved by flipping the matrix after every iteration.
            flip(offset_map, offset_map, -1);
            isFlipped = !isFlipped;
        }
        if (isFlipped) {
            // Correct orientation if we're still in flipped state.
            flip(offset_map, offset_map, -1);
        }
        _offset_map_pyr[s] = offset_map;
    }
    return _offset_map_pyr[0];
}

void RandomizedPatchMatch::updateOffsetMapEntryIfBetter(Mat &patch, Point &candidate_offset,
                                                        Rect &candidate_rect, Mat &source_img, Vec3f *offset_map_entry) {
    // Check if it's fully inside, only try to update then
    Rect source_img_rect(Point(0,0), source_img.size());
    if ((candidate_rect & source_img_rect) == candidate_rect) {
        Mat candidate_patch = source_img(candidate_rect);
        float ssd_value = (float) ssd(patch, candidate_patch);
        if (ssd_value < offset_map_entry->val[2]) {
            offset_map_entry->val[0] = candidate_offset.x;
            offset_map_entry->val[1] = candidate_offset.y;
            offset_map_entry->val[2] = ssd_value;
        }
    }

}

void RandomizedPatchMatch::initializeWithRandomOffsets(Mat &source_img, Mat &target_img, Mat &offset_map) {
    // Seed random;
    srand(target_img.rows * target_img.cols);
    offset_map.create(target_img.rows - _patch_size, target_img.cols - _patch_size, CV_32FC3);
    for (int x = 0; x < offset_map.cols; x++) {
        for (int y = 0; y < offset_map.rows; y++) {
            // Choose offset carfully, so resulting point (when added to current coordinate), is not outside image.
            int randomX = (rand() % (source_img.cols - _patch_size)) - x;
            int randomY = (rand() % (source_img.rows - _patch_size)) - y;

            // TODO: Refactor this to store at every point [Point, double]
            Rect currentPatchRect(x, y, _patch_size, _patch_size);
            Mat currentPatch = target_img(currentPatchRect);
            Mat randomPatch = source_img(Rect(x + randomX, y + randomY, _patch_size, _patch_size));
            float initalSsd = (float) ssd(currentPatch, randomPatch);
            offset_map.at<Vec3f>(y, x) = Vec3f(randomX, randomY, initalSsd);
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
                angle += M_PI * 2;
            angles.at<float>(y, x) = angle / (M_PI * 2) * 360;
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
    imwrite("xoffsets" + filename_modifier + ".exr", out[0]);
    imwrite("yoffsets" + filename_modifier + ".exr", out[1]);
    imwrite("min_dist_img" + filename_modifier + ".exr", out[2]);
    normalize(out[2], normed, 0, 1, cv::NORM_MINMAX, CV_32FC1, Mat() );
    imwrite("min_dist_img_normalized" + filename_modifier + ".exr", normed);
}

int RandomizedPatchMatch::findNumberScales(Mat &source, Mat &target, int patch_size) const {
    int min_dimension = std::min(std::min(std::min(source.cols, source.rows), target.cols), target.rows);
    return (int) log2( min_dimension/ (2.f * patch_size));
}