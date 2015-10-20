//
// Created by panmari on 02.10.15.
//

#ifndef PATCHMATCH_RANDOMIZEDPATCHMATCH_H
#define PATCHMATCH_RANDOMIZEDPATCHMATCH_H

#include "opencv2/imgproc/imgproc.hpp"
#include "PatchMatchProvider.h"
#include "../OffsetMap.h"

class RandomizedPatchMatch : public PatchMatchProvider {

public:
    RandomizedPatchMatch(const cv::Mat &source, const cv::Mat &target, int patch_size, float lambda = 0.5f);
    OffsetMap match() override;

    /* Finds number of scales. At minimum scale, both source & target should still be larger than 2 * patch_size in
     * their minimal dimension.
     */
    int findNumberScales(const cv::Mat &source, const cv::Mat &target, int patch_size) const;

    const cv::Mat getSourceGradientX() const { return _source_grad_x_pyr[0]; };
    const cv::Mat getSourceGradientY() const { return _source_grad_y_pyr[0]; };

private:
    std::vector<cv::Mat> _source_pyr, _target_pyr;
    std::vector<OffsetMap> _offset_map_pyr;

    /**
     * Gradients
     */
    std::vector<cv::Mat> _source_grad_x_pyr, _source_grad_y_pyr, _target_grad_x_pyr, _target_grad_y_pyr;
    std::vector<cv::Rect> _source_rect_pyr;
    const int _patch_size, _max_search_radius;
    // Minimum size image in pyramid is 2x patchSize of lower dimension (or larger).
    const int _nr_scales;

    /**
     * Weight of gradient in distance measure, should be in [0, 1]. Default is 0.5.
     */
    const float _lambda;

    /* Mainly for debugging, dumps offset map to file. */
    void dumpOffsetMapToFile(cv::Mat &offset_map, cv::String filename_modifier) const;

    /*
     * Every entry at offset_map is set to a random & valid (i. e. patch it's pointing to is inside image) offset.
     * Also the corresponding SSD is computed.
     */
    void initializeWithRandomOffsets(const cv::Mat &target_img, const cv::Mat &source_img, const int scale,
                                     OffsetMap &offset_map) const;

    /**
     * Updates 'offset_map_entry' with the given 'candidate_offset' if the patch corresponding to 'candidate_rect' on
     * 'source_img' is a better match than for the given 'patch'.
     */
    void updateOffsetMapEntryIfBetter(const cv::Rect &target_rect, const cv::Point &candidate_offset,
                                      const cv::Rect &candiadate_rect, const int scale,
                                      OffsetMapEntry *offset_map_entry) const;
    float patchDistance(const cv::Rect &source_rect, const cv::Rect &target_rect, const int scale,
                        const float previous_dist = INFINITY) const;
};

#endif //PATCHMATCH_RANDOMIZEDPATCHMATCH_H
