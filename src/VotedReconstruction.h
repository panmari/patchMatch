#ifndef PATCHMATCH_VOTEDRECONSTRUCTION_H
#define PATCHMATCH_VOTEDRECONSTRUCTION_H

#include "opencv2/imgproc/imgproc.hpp"
#include "OffsetMap.h"

class VotedReconstruction {

public:
    /**
     * Assumes an offset map to be a 3 channel image, with the first channel being the x-offset,
     * the y-channel being the y-offset.
     * The patch image is assumed to be the one referenced in offset_map.
     */
    VotedReconstruction(const OffsetMap &offset_map, const cv::Mat &source, const cv::Mat &source_grad_x,
                        const cv::Mat &source_grad_y, int patch_size);

    void reconstruct(cv::Mat &reconstructed) const;

private:
    const cv::Mat _source, _source_grad_x, _source_grad_y;
    OffsetMap _offset_map;
    int _patch_size;

};


#endif //PATCHMATCH_VOTEDRECONSTRUCTION_H
