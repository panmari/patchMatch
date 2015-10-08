//
// Created by moser on 02.10.15.
//

#include "RandomizedPatchMatch.h"
#include "opencv2/highgui/highgui.hpp"
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

const int ITERATIONS_PER_SCALE = 5;
const bool NORMALIZED_DISTANCE = false;
const bool RANDOM_SEARCH = true;
const float ALPHA = 0.5; // Used to modify random search radius. Higher alpha means more random searches.

RandomizedPatchMatch::RandomizedPatchMatch(cv::Mat &img, cv::Mat &img2, int patchSize) :
        _patchSize(patchSize), _max_sarch_radius(max(img2.cols, img2.rows)),
        _nr_scales((int) log2(std::min(img.cols, img.rows) / (2.f * patchSize))) {
    buildPyramid(img, _img_pyr, _nr_scales);
    buildPyramid(img2, _img2_pyr, _nr_scales);
}

cv::Mat RandomizedPatchMatch::match() {
    RNG rng( 0xFFFFFFFF );

    for (int s = _nr_scales - 1; s >= 0; s--) {
        Mat img = _img_pyr[s];
        Mat img2 = _img2_pyr[s];
        Mat offset_map;
        initializeWithRandomOffsets(img, img2, offset_map);
        bool isFlipped = false;
        for (int i = 0; i < ITERATIONS_PER_SCALE; i++) {
            // After half the iterations, merge the lower resolution offset where they're better.
            // This has to be done in an 'even' iteration because of the flipping.
            if (s != _nr_scales - 1 && i == ITERATIONS_PER_SCALE / 2) {
                // TODO: implement merging.
            }
            for (int x = 0; x < offset_map.cols; x++) {
                for (int y = 0; y < offset_map.rows; y++) {
                    Vec3f offset_map_entry = offset_map.at<Vec3f>(y, x);

                    // If image is flipped, we need to get x and y coordinates unflipped for getting the right offset.
                    int x_unflipped, y_unflipped;
                    if (isFlipped) {
                        x_unflipped = offset_map.cols - 1 - x;
                        y_unflipped = offset_map.rows - 1 - y;
                    } else {
                        x_unflipped = x;
                        y_unflipped = y;
                    }
                    Rect currentPatchRect(x_unflipped, y_unflipped, _patchSize, _patchSize);
                    Mat currentPatch = img(currentPatchRect);

                    if (x > 0) {
                        Vec3f offsetLeft = offset_map.at<Vec3f>(y, x - 1);
                        Rect rectLeft((int) offsetLeft[0] + x_unflipped, (int) offsetLeft[1] + y_unflipped,
                                      _patchSize, _patchSize);
                        Point offsetLeftPoint(offsetLeft[0], offsetLeft[1]);
                        updateOffsetMapEntryIfBetter(currentPatch, offsetLeftPoint, rectLeft, img2, &offset_map_entry);
                    }
                    if (y > 0) {
                        Vec3f offsetUp = offset_map.at<Vec3f>(y - 1, x);
                        Rect rectUp((int) offsetUp[0] + x_unflipped, (int) offsetUp[1] + y_unflipped,
                                    _patchSize, _patchSize);
                        Point offsetUpPoint(offsetUp[0], offsetUp[1]);
                        updateOffsetMapEntryIfBetter(currentPatch, offsetUpPoint, rectUp, img2, &offset_map_entry);
                    }

                    Point current_offset(offset_map_entry[0], offset_map_entry[1]);

                    if (RANDOM_SEARCH) {
                        float current_search_radius = _max_sarch_radius;
                        while (current_search_radius > 1) {
                            Point random_point = Point(rng.uniform(-1.f, 1.f) * current_search_radius,
                                                       rng.uniform(-1.f, 1.f) * current_search_radius);
                            Point random_offset = current_offset + random_point;
                            Rect random_rect(x_unflipped + random_offset.x,
                                             y_unflipped + random_offset.y, _patchSize, _patchSize);

                            updateOffsetMapEntryIfBetter(currentPatch, random_offset, random_rect, img2,
                                                         &offset_map_entry);

                            current_search_radius *= ALPHA;
                        }
                    }
                    // Write back newly calculated offsetgit stat_map_entry.
                    offset_map.at<Vec3f>(y, x) = offset_map_entry;
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
        _offset_map = offset_map;
    }
    return _offset_map;
}

void RandomizedPatchMatch::updateOffsetMapEntryIfBetter(Mat &patch, Point &candidate_offset,
                                                        Rect &candidate_rect, Mat &other_img, Vec3f *offset_map_entry) {
    // Check if it's fully inside, only try to update then
    Rect other_img_rect(Point(0,0), other_img.size());
    if ((candidate_rect & other_img_rect) == candidate_rect) {
        Mat candidate_patch = other_img(candidate_rect);
        float ssd_value = (float) ssd(patch, candidate_patch);
        if (ssd_value < offset_map_entry->val[2]) {
            offset_map_entry->val[0] = candidate_offset.x;
            offset_map_entry->val[1] = candidate_offset.y;
            offset_map_entry->val[2] = ssd_value;
        }
    }

}

double RandomizedPatchMatch::ssd(cv::Mat &patch, cv::Mat &patch2) const {
    Mat tmp = patch.clone();
    addWeighted(patch, 1, patch2, -1, 0, tmp);
    Mat squares = tmp.mul(tmp);
    Scalar ssd_channels = sum(squares);
    double ssd = ssd_channels[0] + ssd_channels[1] + ssd_channels[2];
    if (NORMALIZED_DISTANCE) {
        patch.copyTo(tmp);
        patch.mul(tmp);
        Scalar squares_patch = sum(tmp);

        patch2.copyTo(tmp);
        patch2.mul(tmp);
        Scalar squares_patch2 = sum(tmp);
        double normalization = sqrt((squares_patch[0] + squares_patch[1] + squares_patch[2]) *
                                    (squares_patch2[0] + squares_patch2[1] + squares_patch2[2]));
        ssd /= normalization;
    }
    return ssd;
}


void RandomizedPatchMatch::initializeWithRandomOffsets(Mat &img, Mat &img2, Mat &offset_map) {
    // Seed random;
    srand(img.rows * img.cols);
    offset_map.create(img.rows - _patchSize, img.cols - _patchSize, CV_32FC3);
    for (int x = 0; x < offset_map.cols; x++) {
        for (int y = 0; y < offset_map.rows; y++) {
            // Choose offset carfully, so resulting point (when added to current coordinate), is not outside image.
            int randomX = (rand() % offset_map.cols) - x;
            int randomY = (rand() % offset_map.rows) - y;

            // TODO: Refactor this to store at every point [Point, double]
            Rect currentPatchRect(x, y, _patchSize, _patchSize);
            Mat currentPatch = img(currentPatchRect);
            Mat randomPatch = img2(Rect(x + randomX, y + randomY, _patchSize, _patchSize));
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
    std::cout << sum(out[2]) << std::endl;
    imwrite("min_dist_img" + filename_modifier + ".exr", out[2]);
    normalize(out[2], normed, 0, 1, cv::NORM_MINMAX, CV_32FC1, Mat() );
    imwrite("min_dist_img_normalized" + filename_modifier + ".exr", normed);
}