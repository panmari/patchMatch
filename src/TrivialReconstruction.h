#ifndef PATCHMATCH_TRIVIALRECONSTRUCTION_H
#define PATCHMATCH_TRIVIALRECONSTRUCTION_H

#include "opencv2/imgproc/imgproc.hpp"
#include "OffsetMap.h"

/**
 * Trivially reconstructs an image by just taking into account only one single pixel of the closest matching patch.
 */
class TrivialReconstruction {

public:
    /**
     * Assumes an offset map to be a 3 channel image, with the first channel being the x-offset,
     * the y-channel being the y-offset.
     * The patch image is assumed to be the one referenced in offset_map.
     */
    TrivialReconstruction(const OffsetMap *offset_map, const cv::Mat &source_img, const int patch_size);
    cv::Mat reconstruct() const;
private:
    const OffsetMap *_offset_map;
    const cv::Mat _source_img;
    int _patch_size;
};


#endif //PATCHMATCH_TRIVIALRECONSTRUCTION_H
