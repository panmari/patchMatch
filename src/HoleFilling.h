#ifndef PATCHMATCH_HOLEFILLING_H
#define PATCHMATCH_HOLEFILLING_H

#include <memory>
#include <opencv2/imgproc/imgproc.hpp>
#include "OffsetMap.h"

class HoleFilling {

public:
    const static cv::Vec3f hole_color;
    /**
     * @param img the image of which we want to fill the hole of, usually in L*a*b* color space.
     * @param hole a bitmask of the hole, non-zero where the hole is, zero otherwise (one channel uint8).
     * @param patch_size the sizes of the patches to be used. A useful default is 7.
     */
    HoleFilling(const cv::Mat &img, const cv::Mat &hole, int patch_size);

    /**
     * Returns a the full image with the hole inpainted. Has the same color space as the image given in construction.
     */
    cv::Mat run();
    cv::Mat solutionFor(const int scale) const;

    std::vector<cv::Mat> _img_pyr, _hole_pyr, _target_area_pyr;
    std::vector<std::shared_ptr<OffsetMap>> _offset_map_pyr;
    std::vector<cv::Rect> _target_rect_pyr;
    int _nr_scales;
private:
    const int _patch_size;
    void upscaleSolution(const int current_scale, const std::vector<cv::Mat> &rotated_sources,
                            cv::Mat &upscaled_solution) const;
    cv::Rect computeTargetRect(const cv::Mat &img, const cv::Mat &hole, int patch_size) const;
};

#endif //PATCHMATCH_HOLEFILLING_H
