//
// Created by moser on 06.10.15.
//

#ifndef PATCHMATCH_VOTEDRECONSTRUCTION_H
#define PATCHMATCH_VOTEDRECONSTRUCTION_H

#include "opencv2/imgproc/imgproc.hpp"

class VotedReconstruction {

public:
    /**
     * Assumes an offset map to be a 3 channel image, with the first channel being the x-offset,
     * the y-channel being the y-offset.
     * The patch image is assumed to be the one referenced in offset_map.
     */
    VotedReconstruction(cv::Mat &offset_map, cv::Mat &source, int patch_size);

    cv::Mat reconstruct() const;

private:
    cv::Mat _offset_map, _source;
    int _patch_size;

};


#endif //PATCHMATCH_VOTEDRECONSTRUCTION_H
