//
// Created by moser on 02.10.15.
//

#include "RandomizedPatchMatch.h"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using cv::addWeighted;
using cv::flip;
using cv::Mat;
using cv::Point;
using cv::Scalar;
using cv::String;
using cv::Rect;
using cv::RNG;
using cv::Vec3f;

using std::max;

const int MAX_ITERATIONS = 5;
const bool NORMALIZED_DISTANCE = false;
const float ALPHA = 0.5;

RandomizedPatchMatch::RandomizedPatchMatch(cv::Mat &img, cv::Mat &img2, int patchSize) :
        _img(img), _img2(img2), _patchSize(patchSize), _rect_full_img2(Point(0,0), _img2.size()),
        _max_sarch_radius(max(img2.cols, img2.rows)){
    initializeOffsets(patchSize);
}

cv::Mat RandomizedPatchMatch::match() {
    bool isFlipped = false;

    RNG rng( 0xFFFFFFFF );

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        for (int x = 0; x < _img.cols - _patchSize; x++) {
            for (int y = 0; y < _img.rows - _patchSize; y++) {
                Vec3f offset_map_entry = _offset_map.at<Vec3f>(y, x);

                // If image is flipped, we need to get x and y coordinates unflipped for getting the right offset.
                int x_unflipped, y_unflipped;
                if (isFlipped) {
                    x_unflipped = _offset_map.cols - x;
                    y_unflipped = _offset_map.rows - y;
                } else {
                    x_unflipped = x;
                    y_unflipped = y;
                }
                Rect currentPatchRect(x_unflipped, y_unflipped, _patchSize, _patchSize);
                Mat currentPatch = _img(currentPatchRect);

                if (x > 0) {
                    Vec3f offsetLeft = _offset_map.at<Vec3f>(y, x - 1);
                    Rect rectLeft((int) offsetLeft[0] + x_unflipped, (int) offsetLeft[1] + y_unflipped,
                                  _patchSize, _patchSize);
                    Point offsetLeftPoint(offsetLeft[0], offsetLeft[1]);
                    updateOffsetMapEntryIfBetter(currentPatch, offsetLeftPoint, rectLeft, &offset_map_entry);
                }
                if (y > 0) {
                    Vec3f offsetUp = _offset_map.at<Vec3f>(y - 1, x);
                    Rect rectUp((int) offsetUp[0] + x_unflipped, (int) offsetUp[1] + y_unflipped,
                                _patchSize, _patchSize);
                    Point offsetUpPoint(offsetUp[0], offsetUp[1]);
                    updateOffsetMapEntryIfBetter(currentPatch, offsetUpPoint, rectUp, &offset_map_entry);
                }

                Point current_offset(offset_map_entry[0], offset_map_entry[1]);

                float current_search_radius = _max_sarch_radius;
                while (current_search_radius > 1) {
                    Point random_point = Point(rng.uniform(-1, 1), rng.uniform(-1, 1));
                    Point random_offset = current_offset + random_point * current_search_radius;
                    Rect random_rect(random_offset.x, random_offset.y, _patchSize, _patchSize);

                    updateOffsetMapEntryIfBetter(currentPatch, random_offset, random_rect, &offset_map_entry);

                    current_search_radius *= ALPHA;
                }

                // Write back newly calculated offsetgit stat_map_entry.
                _offset_map.at<Vec3f>(y, x) = offset_map_entry;
            }
        }
        dumpOffsetMapToFile(std::to_string(i));
        // Every second iteration, we go the other way round (start at bottom, propagate from right and down).
        // This effect can be achieved by flipping the matrix after every iteration.
        flip(_offset_map, _offset_map, -1);
        isFlipped = !isFlipped;
    }
    if (isFlipped) {
        // Correct orientation if we're still in flipped state.
        flip(_offset_map, _offset_map, -1);
    }
    return _offset_map;
}

void RandomizedPatchMatch::updateOffsetMapEntryIfBetter(Mat &patch, Point &candidate_offset,
                                                        Rect &candidate_rect, Vec3f *offset_map_entry) {
    if ((candidate_rect & _rect_full_img2) == candidate_rect) { // Check if it's fully inside
        Mat candidate_patch = _img2(candidate_rect);
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


void RandomizedPatchMatch::initializeOffsets(int patchSize) {
    // Seed random;
    srand(42);
    _offset_map.create(_img.rows - patchSize, _img.cols - patchSize, CV_32FC3);
    for (int x = 0; x < _offset_map.cols; x++) {
        for (int y = 0; y < _offset_map.rows; y++) {
            int randomX = (rand() % _offset_map.cols) - x;
            int randomY = (rand() % _offset_map.rows) - y;

            // TODO: Refactor this to store at every point [Point, double]
            Rect currentPatchRect(x, y, patchSize, patchSize);
            Mat currentPatch = _img(currentPatchRect);
            Mat randomPatch = _img2(Rect(x + randomX, y + randomY, patchSize, patchSize));
            float initalSsd = (float) ssd(currentPatch, randomPatch);
            _offset_map.at<Vec3f>(y, x) = Vec3f(randomX, randomY, initalSsd);
        }
    }

}

void RandomizedPatchMatch::dumpOffsetMapToFile(String filename_modifier) const {
    Mat xoffsets, yoffsets, diff;
    Mat out[] = {xoffsets, yoffsets, diff};
    split(_offset_map, out);
    Mat angles = Mat::zeros(_offset_map.size(), CV_32FC1);
    Mat magnitudes = Mat::zeros(_offset_map.size(), CV_32FC1);

    // Produce some nice to look at output by coding angle to best patch as hue, magnitude as saturation.
    for (int x = 0; x < _offset_map.cols; x++) {
        for (int y = 0; y < _offset_map.rows; y++) {
            Vec3f offset_map_entry = _offset_map.at<Vec3f>(y, x);
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
    Mat hsv_array[] = {angles, magnitudes, Mat::ones(_offset_map.size(), CV_32FC1)};
    Mat hsv;
    cv::merge(hsv_array, 3, hsv);
    cvtColor(hsv, hsv, CV_HSV2BGR);
    imwrite("hsv_offsets" + filename_modifier + ".exr", hsv);

    // Dump unnormalized values for inspection.
    imwrite("xoffsets" + filename_modifier + ".exr", out[0]);
    imwrite("yoffsets" + filename_modifier + ".exr", out[1]);
    std::cout << sum(out[2]) << std::endl;
    imwrite("minDistImg" + filename_modifier + ".exr", out[2]);
}
