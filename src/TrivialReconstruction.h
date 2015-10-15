#ifndef PATCHMATCH_TRIVIALRECONSTRUCTION_H
#define PATCHMATCH_TRIVIALRECONSTRUCTION_H

#include "opencv2/imgproc/imgproc.hpp"

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
    TrivialReconstruction(cv::Mat &offset_map, cv::Mat &source_img, int patch_size);
    cv::Mat reconstruct() const;
private:
    cv::Mat _offset_map, _source_img;
    int _patch_size;
};


#endif //PATCHMATCH_TRIVIALRECONSTRUCTION_H
