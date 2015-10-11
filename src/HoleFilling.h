#ifndef PATCHMATCH_HOLEFILLING_H
#define PATCHMATCH_HOLEFILLING_H

#include "opencv2/imgproc/imgproc.hpp"

class HoleFilling {

public:
    HoleFilling(cv::Mat &img, cv::Mat &hole, int patch_size);
    cv::Rect computeTargetRect(cv::Mat &img, cv::Mat &hole, int patch_size) const;
    cv::Mat run();

    std::vector<cv::Mat> _img_pyr, _hole_pyr, _target_area_pyr;
    int _nr_scales;
private:
    cv::Mat _img, _hole;
    const int _patch_size;
};

#endif //PATCHMATCH_HOLEFILLING_H
