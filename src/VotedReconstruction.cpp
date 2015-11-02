#include "VotedReconstruction.h"
#include "PoissonSolver.h"
#include "util.h"
#include <iostream>

using cv::COLOR_GRAY2BGR;
using cv::divide;
using cv::Mat;
using cv::meanStdDev;
using cv::Rect;
using cv::Size;
using cv::Vec3f;
using pmutil::naiveMeanShift;
using std::shared_ptr;
using std::vector;

/**
 * Patches with higher similarity have higher weights in reconstruction. If false, ever patch has the same weight (1).
 */
const bool WEIGHTED_BY_SIMILARITY = true;

namespace {
    class ParallelModeAwareReconstruction : public cv::ParallelLoopBody {
    private:
        const vector<vector<Vec3f>> _colors;
        const vector<vector<float>> _weights;
        const float _mean_shift_bandwith_scale;
        Mat *_reconstructed_flat;

    public:
        ParallelModeAwareReconstruction(const vector<vector<Vec3f>> &colors, const vector<vector<float>> &weights,
                                        const float mean_shift_bandwith_scale, Mat *reconstructed_flat)
                : _colors(colors), _weights(weights), _mean_shift_bandwith_scale(mean_shift_bandwith_scale),
                  _reconstructed_flat(reconstructed_flat) { }

        virtual void operator()(const cv::Range &r) const {
            for (int i = r.start; i < r.end; i++) {
                const vector<Vec3f> one_pixel_colors = _colors[i];
                if (one_pixel_colors.empty())
                    continue;
                Scalar mean, std;
                meanStdDev(one_pixel_colors, mean, std);
                float avg_std_channels = static_cast<float>(std[0] + std[1] + std[2]) / 3;
                float sigma_mean_shift =  avg_std_channels * _mean_shift_bandwith_scale;

                if (avg_std_channels > 0.1f) {
                    vector<Vec3f> modes;
                    vector<int> mode_assignments;
                    naiveMeanShift(one_pixel_colors, sigma_mean_shift, &modes, &mode_assignments);
                    vector<int> occurrences(modes.size(), 0);
                    for (int assignment: mode_assignments) {
                        occurrences[assignment]++;
                    }
                    auto max_occurences_iter = std::max_element(occurrences.begin(), occurrences.end());
                    long max_mode = std::distance(occurrences.begin(), max_occurences_iter);
                    const vector<float> one_pixel_weights = _weights[i];
                    Vec3f final_color(0, 0, 0);
                    double total_weight = 0;
                    for (int color_idx = 0; color_idx < one_pixel_colors.size(); color_idx++) {
                        if (mode_assignments[color_idx] == max_mode) {
                            float weight = one_pixel_weights[color_idx];
                            final_color += one_pixel_colors[color_idx] * weight;
                            total_weight += weight;
                        }
                    }
                    _reconstructed_flat->at<Vec3f>(i) = final_color / total_weight;
                } else {
                    // If there is not much variance, there is no need to do voting, simply take first color.
                    _reconstructed_flat->at<Vec3f>(i) = one_pixel_colors[0];
                }
            }
        }
    };
}

VotedReconstruction::VotedReconstruction(const shared_ptr<OffsetMap> offset_map, const Mat &source,
                                         const Mat &hole, int patch_size, int scale) :
        _offset_map(offset_map), _source(source), _hole(hole), _patch_size(patch_size), _scale(scale) {
    if (scale != 1) {
        // Source images need some border for reconstruction if we're using bigger patches.
        copyMakeBorder(source, _source, 0, 1, 0, 1, cv::BORDER_REFLECT);
    }
}

void VotedReconstruction::reconstruct(Mat &reconstructed, float mean_shift_bandwith_scale) const {
    Size reconstructed_size((_offset_map->_width - 1 + _patch_size) * _scale,
                            (_offset_map->_height - 1 + _patch_size) * _scale);
    reconstructed = Mat::zeros(reconstructed_size, CV_32FC3);

    // Wexler et al suggest using the 75 percentile of the distances as sigma.
    const float sigma = _offset_map->get75PercentileDistance();
    const float two_sigma_sqr = sigma * sigma * 2;
    vector<vector<Vec3f>> colors(reconstructed_size.width * reconstructed_size.height);
    vector<vector<float>> weights(reconstructed_size.width * reconstructed_size.height);
    for (int x = 0; x < _offset_map->_width; x++) {
        for (int y = 0; y < _offset_map->_height; y++) {
            // Go over all patches that contain this image.
            const OffsetMapEntry offset_map_entry = _offset_map->at(y, x);
            const Point offset = offset_map_entry.offset;
            // Get image data of matching patch
            Rect matching_patch_rect((x + offset.x) * _scale, (y + offset.y) * _scale,
                                     _patch_size * _scale, _patch_size * _scale);
            Mat matching_patch = _source(matching_patch_rect);

            float weight;
            if (WEIGHTED_BY_SIMILARITY) {
                float normalized_dist = sqrtf(offset_map_entry.distance);
                weight = expf(-normalized_dist / two_sigma_sqr);
            }
            else {
                weight = 1;
            }

            for (int x_patch = 0; x_patch < _patch_size * _scale; x_patch++) {
                for (int y_patch = 0; y_patch < _patch_size * _scale; y_patch++) {
                    Point p(x * _scale + x_patch, y * _scale + y_patch);
                    if (_hole.at<uchar>(p) != 0) {
                        int idx = p.x + reconstructed_size.width * p.y;
                        colors[idx].push_back(matching_patch.at<Vec3f>(y_patch, x_patch));
                        weights[idx].push_back(weight);
                    }
                }
            }
        }
    }
    Mat reconstructed_flat = reconstructed.reshape(3, 1);
    parallel_for_(cv::Range(0, reconstructed_flat.cols),
                  ParallelModeAwareReconstruction(colors, weights, mean_shift_bandwith_scale, &reconstructed_flat));
}

