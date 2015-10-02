//
// Created by moser on 02.10.15.
//

#include "RandomizedPatchMatch.h"
#include "opencv2/highgui/highgui.hpp"

using cv::addWeighted;
using cv::flip;
using cv::Mat;
using cv::Point;
using cv::Scalar;
using cv::Rect;
using cv::Vec3f;

const int MAX_ITERATIONS = 5;

RandomizedPatchMatch::RandomizedPatchMatch(cv::Mat &img, cv::Mat &img2) : _img(img), _img2(img2) {
}

cv::Mat RandomizedPatchMatch::match(int patchSize) {
    initializeOffsets(patchSize);

    Rect rectFullImg(Point(0,0), _img2.size());

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        for (int x = 0; x < _img.cols - patchSize; x++) {
            for (int y = 0; y < _img.rows - patchSize; y++) {
                Rect currentPatchRect(x, y, patchSize, patchSize);
                Mat currentPatch = _img(currentPatchRect);
                Vec3f currentOffsetEntry = _offset_map.at<Vec3f>(y, x);

                if (x > 0) {
                    Vec3f offsetLeft = _offset_map.at<Vec3f>(y, x - 1);
                    Rect rectLeft((int) offsetLeft[0] + x, (int) offsetLeft[1] + y, patchSize, patchSize);
                    if ((rectLeft & rectFullImg) == rectLeft) {
                        Mat matchingPatchLeft = _img2(rectLeft);
                        float leftSsd = (float) ssd(currentPatch, matchingPatchLeft);
                        if (leftSsd < currentOffsetEntry[2]) {
                            _offset_map.at<Vec3f>(y, x) = Vec3f(offsetLeft[0], offsetLeft[1], leftSsd);
                        }
                    }
                }
                if (y > 0) {
                    Vec3f offsetUp = _offset_map.at<Vec3f>(y - 1, x);
                    Rect rectUp((int) offsetUp[0] + x, (int) offsetUp[1] + y, patchSize, patchSize);
                    if ((rectUp & rectFullImg) == rectUp) { // Check if it's fully inside
                        Mat matchingPatchUp = _img2(rectUp);
                        float upSsd = (float) ssd(currentPatch, matchingPatchUp);
                        if (upSsd < currentOffsetEntry[2]) {
                            _offset_map.at<Vec3f>(y, x) = Vec3f(offsetUp[0], offsetUp[1], upSsd);
                        }
                    }
                }
            }
        }
        // Every second iteration, we go the other way round (start at bottom, propagate from right and down).
        // This effect can be achieved by flipping the matrix after every iteration.
        flip(_offset_map, _offset_map, -1);
    }
    if (MAX_ITERATIONS % 2 == 1) {
        // Correct orientation based on numbers of iterations.
        flip(_offset_map, _offset_map, -1);
    }
    return _offset_map;
}

double RandomizedPatchMatch::ssd(cv::Mat &patch, cv::Mat &patch2) const {
    Mat tmp;
    addWeighted(patch, 1, patch2, -1, 0, tmp);
    tmp.mul(tmp);
    Scalar ssd_channels = sum(tmp);

    patch.copyTo(tmp);
    patch.mul(tmp);
    Scalar squares_patch = sum(tmp);

    patch2.copyTo(tmp);
    patch2.mul(tmp);
    Scalar squares_patch2 = sum(tmp);
    double normalization = sqrt((squares_patch[0] + squares_patch[1] + squares_patch[2]) *
                                        (squares_patch2[0] + squares_patch2[1] + squares_patch2[2]));
    return (ssd_channels[0] + ssd_channels[1] + ssd_channels[2])/normalization;
}

void RandomizedPatchMatch::initializeOffsets(int patchSize) {
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
