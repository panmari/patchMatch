//
// Created by moser on 06.10.15.
//

#ifndef PATCHMATCH_TRIVIALRECONSTRUCTION_H
#define PATCHMATCH_TRIVIALRECONSTRUCTION_H

#include "opencv2/imgproc/imgproc.hpp"

class TrivialReconstruction {

public:
    TrivialReconstruction(cv::Mat &offset_map);

private:
    cv::Mat _offset_map;
};


#endif //PATCHMATCH_TRIVIALRECONSTRUCTION_H
