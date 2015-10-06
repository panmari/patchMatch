#include "TrivialReconstruction.h"
#include <iostream>

using cv::Mat;
using cv::Vec3f;

TrivialReconstruction::TrivialReconstruction(Mat &offset_map, Mat &patch_img) :
        _offset_map(offset_map), _patch_img(patch_img) { }

Mat TrivialReconstruction::reconstruct() const {
    Mat reconstructed;
    reconstructed.create(_offset_map.size(), CV_32FC3);
    std::cout << "Size of image: " << _patch_img.size() << std::endl;
    for (int x = 0; x < _offset_map.cols; x++) {
        for (int y = 0; y < _offset_map.rows; y++) {
            Vec3f offset_map_entry = _offset_map.at<Vec3f>(y, x);
            int match_x = x + offset_map_entry[0];
            int match_y = y + offset_map_entry[1];

            if (match_x < 0 || match_x >= _patch_img.cols || match_y < 0 || match_y >= _patch_img.rows) {
                std::cout << "Not inside image: (" << match_x << "," << match_y << ")" << std::endl;
            }

            Vec3f best_match = _patch_img.at<Vec3f>(match_y, match_x);
            reconstructed.at<Vec3f>(y, x) = best_match;
        }
    }

    cvtColor(reconstructed, reconstructed, CV_Lab2BGR);
    return reconstructed;
}