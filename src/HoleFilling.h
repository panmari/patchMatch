#ifndef PATCHMATCH_HOLEFILLING_H
#define PATCHMATCH_HOLEFILLING_H

#include "opencv2/imgproc/imgproc.hpp"

class HoleFilling {

public:
    HoleFilling(cv::Mat &img, cv::Mat &hole, int patch_size);

    cv::Mat _target_area;
    cv::Rect _target_rect;
private:
    cv::Mat _img, _hole;
    const int _patch_size;
};


#endif //PATCHMATCH_HOLEFILLING_H
