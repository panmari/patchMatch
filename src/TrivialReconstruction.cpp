//
// Created by moser on 06.10.15.
//

#include "TrivialReconstruction.h"

using cv::Mat;
using cv::Vec3f;

TrivialReconstruction::TrivialReconstruction(Mat &offset_map, Mat &patch_img) :
        _offset_map(offset_map), _patch_img(patch_img) { }

Mat TrivialReconstruction::reconstruct() const {
    Mat reconstructed;
    reconstructed.create(_offset_map.size(), CV_32FC3);
    for (int x = 0; x < _offset_map.cols; x++) {
        for (int y = 0; y < _offset_map.rows; y++) {
            Vec3f offset_map_entry = _offset_map.at<Vec3f>(y, x);
            Vec3f best_match = _patch_img.at<Vec3f>(y + offset_map_entry[1], x + offset_map_entry[0]);
            reconstructed.at<Vec3f>(y, x) = best_match;
        }
    }

    cvtColor(reconstructed, reconstructed, CV_Lab2BGR);
    return reconstructed;
}