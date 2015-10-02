//
// Created by panmari on 02.10.15.
//

#ifndef PATCHMATCH_RANDOMIZEDPATCHMATCH_H
#define PATCHMATCH_RANDOMIZEDPATCHMATCH_H

#include "opencv2/imgproc/imgproc.hpp"

class RandomizedPatchMatch {

public:
    RandomizedPatchMatch(cv::Mat &img, cv::Mat &img2);
    cv::Mat match(int patchSize);

private:
    const cv::Mat _img, _img2;
    cv::Mat _offset_map;
};


#endif //PATCHMATCH_RANDOMIZEDPATCHMATCH_H
