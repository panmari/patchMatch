#ifndef PATCHMATCH_HOLEFILLING_H
#define PATCHMATCH_HOLEFILLING_H

#include "opencv2/imgproc/imgproc.hpp"

class HoleFilling {

public:
    HoleFilling(cv::Mat &img, cv::Mat &hole, int patch_size);
    cv::Rect computeTargetRect(cv::Mat &img, cv::Mat &hole, int patch_size);

    cv::Mat _target_area;
    cv::Rect _target_rect;
    std::vector<cv::Mat> _img_pyr, _hole_pyr;
private:
    cv::Mat _img, _hole;
    const int _patch_size;
    const int _nr_scales;
};


#endif //PATCHMATCH_HOLEFILLING_H
