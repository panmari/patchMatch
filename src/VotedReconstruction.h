#ifndef PATCHMATCH_VOTEDRECONSTRUCTION_H
#define PATCHMATCH_VOTEDRECONSTRUCTION_H

#include <memory>
#include <opencv2/imgproc/imgproc.hpp>
#include "OffsetMap.h"

class VotedReconstruction {

public:
    /**
     * Assumes an offset map to be a 3 channel image, with the first channel being the x-offset,
     * the y-channel being the y-offset.
     * The patch image is assumed to be the one referenced in offset_map.
     */
    VotedReconstruction(const std::shared_ptr<OffsetMap> offset_map, const cv::Mat &source, const cv::Mat &hole,
                        int patch_size, int scale = 1);

    void reconstruct(cv::Mat &reconstructed, float mean_shift_bandwith_scale) const;

private:
    cv::Mat _source;
    const cv::Mat _hole;
    const std::shared_ptr<OffsetMap> _offset_map;
    const int _patch_size, _scale;

};


#endif //PATCHMATCH_VOTEDRECONSTRUCTION_H
