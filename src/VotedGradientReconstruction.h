#ifndef PATCHMATCH_VOTEDGRADIENTRECONSTRUCTION_H
#define PATCHMATCH_VOTEDGRADIENTRECONSTRUCTION_H

#include <memory>
#include <opencv2/imgproc/imgproc.hpp>
#include "OffsetMap.h"

class VotedGradientReconstruction {

public:
    VotedGradientReconstruction(const std::shared_ptr<OffsetMap> offset_map, const cv::Mat &source,
                                const cv::Mat &source_grad_x, const cv::Mat &source_grad_y, const cv::Mat &hole,
                                const int patch_size, const int scale = 1);

    void reconstruct(cv::Mat &reconstructed,
                     cv::Mat &reconstructed_x_gradient,
                     cv::Mat &reconstructed_y_gradient) const;

private:
    const cv::Mat _source, _source_grad_x, _source_grad_y, _hole;
    const std::shared_ptr<OffsetMap> _offset_map;
    const int _patch_size, _scale;

};


#endif //PATCHMATCH_VOTEDGRADIENTRECONSTRUCTION_H
