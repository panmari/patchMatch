#include "VotedReconstruction.h"

using cv::divide;
using cv::Mat;
using cv::Rect;
using cv::Vec3f;
using std::vector;

VotedReconstruction::VotedReconstruction(Mat &offset_map, Mat &patch_img, int patch_size) :
        _offset_map(offset_map), _patch_img(patch_img), _patch_size(patch_size) { }

Mat VotedReconstruction::reconstruct() const {
    Mat reconstructed = Mat::zeros(_offset_map.rows + _patch_size, _offset_map.cols + _patch_size, CV_32FC3);
    Mat count = Mat::zeros(reconstructed.size(), CV_32FC1);
    for (int x = 0; x < _offset_map.cols; x++) {
        for (int y = 0; y < _offset_map.rows; y++) {
            // Go over all patches that contain this image.
            Vec3f offset_map_entry = _offset_map.at<Vec3f>(y, x);
            int match_x = x + offset_map_entry[0];
            int match_y = y + offset_map_entry[1];
            // Get image data of matching patch
            Rect matching_patch_rect(match_x, match_y, _patch_size, _patch_size);
            Mat matching_patch = _patch_img(matching_patch_rect);

            // Add to all pixels at once.
            Rect current_patch_rect(x, y, _patch_size, _patch_size);
            reconstructed(current_patch_rect) += matching_patch;

            // Remember for every pixel, how many patches were added up for later division.
            count(current_patch_rect) += 1;
        }
    }

    // Divide every channel by count.
    vector<Mat> channels(3);
    split(reconstructed, channels);
    for (Mat chan: channels) {
        divide(chan, count, chan);
    }
    merge(channels, reconstructed);

    cvtColor(reconstructed, reconstructed, CV_Lab2BGR);
    return reconstructed;
}