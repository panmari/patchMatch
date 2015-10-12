#ifndef PATCHMATCH_HOLEFILLING_H
#define PATCHMATCH_HOLEFILLING_H

#include "opencv2/imgproc/imgproc.hpp"

class HoleFilling {

public:
    HoleFilling(cv::Mat &img, cv::Mat &hole, int patch_size);
    cv::Mat run();
    cv::Mat solutionFor(int scale) const;

    std::vector<cv::Mat> _img_pyr, _hole_pyr, _target_area_pyr;
    std::vector<cv::Rect> _target_rect_pyr;
    int _nr_scales;
private:
    cv::Mat _img, _hole;
    const int _patch_size;
    cv::Mat upscaleSolution(int current_scale) const;
    cv::Rect computeTargetRect(cv::Mat &img, cv::Mat &hole, int patch_size) const;
};

#endif //PATCHMATCH_HOLEFILLING_H
