#include "VotedReconstruction.h"

using cv::COLOR_GRAY2BGR;
using cv::divide;
using cv::Mat;
using cv::Rect;
using cv::Vec3f;
using std::vector;

/**
 * Patches with higher similarity have higher weights in reconstruction. If false, ever patch has the same weight (1).
 */
const bool WEIGHTED_BY_SIMILARITY = false;
const float SIGMA_SQR = 1;

VotedReconstruction::VotedReconstruction(const Mat &offset_map, const Mat &source, const Mat &source_grad_x,
                                         const Mat &source_grad_y, int patch_size) :
        _offset_map(offset_map), _source(source), _source_grad_x(source_grad_x), _source_grad_y(source_grad_y),
        _patch_size(patch_size) { }

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
            Mat matching_patch = _source(matching_patch_rect);

            float weight;
            if (WEIGHTED_BY_SIMILARITY) {
                // Apply square root to get L2 distance (kind of), then divide by patchsize.
                float normalized_dist = sqrt(offset_map_entry[2]) / (_patch_size  * _patch_size);
                weight = exp(-normalized_dist / 2 * SIGMA_SQR);
            }
            else {
                weight = 1;
            }
            // Add to all pixels at once.
            Rect current_patch_rect(x, y, _patch_size, _patch_size);

            reconstructed(current_patch_rect) += matching_patch * weight;

            // Remember for every pixel, how many patches were added up for later division.
            count(current_patch_rect) += weight;
        }
    }

    // Divide every channel by count (reproduce counts on 3 channels first).
    Mat weights3d;
    cvtColor(count, weights3d, COLOR_GRAY2BGR);
    divide(reconstructed, weights3d, reconstructed);
    return reconstructed;
}