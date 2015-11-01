#include "VotedGradientReconstruction.h"

using cv::COLOR_GRAY2BGR;
using cv::divide;
using cv::Mat;
using cv::Point;
using cv::Rect;
using cv::Size;
using std::shared_ptr;

VotedGradientReconstruction::VotedGradientReconstruction(const shared_ptr<OffsetMap> offset_map,
                                                         const Mat &source, const Mat &source_grad_x,
                                                         const Mat &source_grad_y, const Mat &hole,
                                                         int patch_size, int scale) :
        _offset_map(offset_map), _source(source), _source_grad_x(source_grad_x), _source_grad_y(source_grad_y),
        _hole(hole), _patch_size(patch_size), _scale(scale) {
    if (scale != 1) {
        // Source images need some border for reconstruction if we're using bigger patches.
        copyMakeBorder(source, _source, 0, 1, 0, 1, cv::BORDER_REFLECT);
        copyMakeBorder(source_grad_x, _source_grad_x, 0, 1, 0, 1, cv::BORDER_REFLECT);
        copyMakeBorder(source_grad_y, _source_grad_y, 0, 1, 0, 1, cv::BORDER_REFLECT);
    }
}


void VotedGradientReconstruction::reconstruct(Mat &reconstructed, Mat &reconstructed_x_gradient,
                                              Mat &reconstructed_y_gradient) const {
    Size reconstructed_size((_offset_map->_width - 1 + _patch_size) * _scale,
                            (_offset_map->_height - 1 + _patch_size) * _scale);
    reconstructed = Mat::zeros(reconstructed_size, CV_32FC3);
    reconstructed_x_gradient = Mat::zeros(reconstructed_size, CV_32FC3);
    reconstructed_y_gradient = Mat::zeros(reconstructed_size, CV_32FC3);

    // Wexler et al suggest using the 75 percentile of the distances as sigma.
    const float sigma = _offset_map->get75PercentileDistance();
    const float two_sigma_sqr = sigma * sigma * 2;
    Mat count = Mat::zeros(reconstructed.size(), CV_32FC1);

    for (int x = 0; x < _offset_map->_width; x++) {
        for (int y = 0; y < _offset_map->_height; y++) {
            // Go over all patches that contain this image.
            const OffsetMapEntry offset_map_entry = _offset_map->at(y, x);
            const Point offset = offset_map_entry.offset;
            // Get image data of matching patch
            Rect matching_patch_rect((x + offset.x) * _scale, (y + offset.y) * _scale,
                                     _patch_size * _scale, _patch_size * _scale);
            Mat matching_patch = _source(matching_patch_rect);
            Mat matching_patch_grad_x = _source_grad_x(matching_patch_rect);
            Mat matching_patch_grad_y = _source_grad_y(matching_patch_rect);

            float normalized_dist = sqrtf(offset_map_entry.distance);
            float weight = expf(-normalized_dist / two_sigma_sqr);
            Rect current_patch_rect(x * _scale, y * _scale, _patch_size * _scale, _patch_size * _scale);
            reconstructed(current_patch_rect) += matching_patch * weight;
            reconstructed_x_gradient(current_patch_rect) += matching_patch_grad_x * weight;
            reconstructed_y_gradient(current_patch_rect) += matching_patch_grad_y * weight;
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
}