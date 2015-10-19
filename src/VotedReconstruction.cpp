#include "VotedReconstruction.h"
#include "PoissonSolver.h"

using cv::COLOR_GRAY2BGR;
using cv::divide;
using cv::Mat;
using cv::Rect;
using cv::Size;
using cv::Vec3f;

/**
 * Patches with higher similarity have higher weights in reconstruction. If false, ever patch has the same weight (1).
 */
const bool WEIGHTED_BY_SIMILARITY = false;
const float SIGMA_SQR = 1;

VotedReconstruction::VotedReconstruction(const Mat &offset_map, const Mat &source, const Mat &source_grad_x,
                                         const Mat &source_grad_y, int patch_size) :
        _offset_map(offset_map), _source(source), _source_grad_x(source_grad_x), _source_grad_y(source_grad_y),
        _patch_size(patch_size) { }

void VotedReconstruction::reconstruct(Mat &reconstructed_solved) const {
    Size reconstructed_size(_offset_map.rows + _patch_size - 1, _offset_map.cols + _patch_size - 1);
    Mat reconstructed = Mat::zeros(reconstructed_size, CV_32FC3);
    Mat reconstructed_x_gradient = Mat::zeros(reconstructed_size, CV_32FC3);
    Mat reconstructed_y_gradient = Mat::zeros(reconstructed_size, CV_32FC3);

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
            Mat matching_patch_grad_x = _source_grad_x(matching_patch_rect);
            Mat matching_patch_grad_y = _source_grad_y(matching_patch_rect);

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
            reconstructed_x_gradient(current_patch_rect) += matching_patch_grad_x;
            reconstructed_y_gradient(current_patch_rect) += matching_patch_grad_y;
            // Remember for every pixel, how many patches were added up for later division.
            count(current_patch_rect) += weight;
        }
    }

    // Divide every channel by count (reproduce counts on 3 channels first).
    Mat weights3d;
    cvtColor(count, weights3d, COLOR_GRAY2BGR);
    divide(reconstructed, weights3d, reconstructed);
    divide(reconstructed_x_gradient, weights3d, reconstructed_x_gradient);
    divide(reconstructed_y_gradient, weights3d, reconstructed_y_gradient);
    PoissonSolver ps(reconstructed, reconstructed_x_gradient, reconstructed_y_gradient);
    ps.solve(reconstructed_solved);
}