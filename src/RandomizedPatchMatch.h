//
// Created by panmari on 02.10.15.
//

#ifndef PATCHMATCH_RANDOMIZEDPATCHMATCH_H
#define PATCHMATCH_RANDOMIZEDPATCHMATCH_H

#include "opencv2/imgproc/imgproc.hpp"

class RandomizedPatchMatch {

public:
    RandomizedPatchMatch(cv::Mat &source, cv::Mat &target, int patch_size);
    cv::Mat match();

    /* Finds number of scales. At minimum scale, both source & target should still be larger than 2 * patch_size in
     * their minimal dimension.
     */
    int findNumberScales(cv::Mat &source, cv::Mat &target, int patch_size) const;

private:
    std::vector<cv::Mat> _source_pyr, _target_pyr, _offset_map_pyr;
    const int _patch_size, _max_search_radius;
    // Minimum size image in pyramid is 2x patchSize of lower dimension (or larger).
    const int _nr_scales;

    /* Mainly for debugging, dumps offset map to file. */
    void dumpOffsetMapToFile(cv::Mat &offset_map, cv::String filename_modifier) const;

    /*
     * Every entry at offset_map is set to a random & valid (i. e. patch it's pointing to is inside image) offset.
     * Also the corresponding SSD is computed.
     */
    void initializeWithRandomOffsets(cv::Mat &target_img, cv::Mat &source_img, cv::Mat &offset_map) const;

    /**
     * Updates 'offset_map_entry' with the given 'candidate_offset' if the patch corresponding to 'candidate_rect' on
     * 'source_img' is a better match than for the given 'patch'.
     */
    void updateOffsetMapEntryIfBetter(cv::Mat &patch, cv::Point &candidate_offset,
                                      cv::Rect &candiadate_rect, cv::Mat &source_img, cv::Vec3f *offset_map_entry) const;

};

#endif //PATCHMATCH_RANDOMIZEDPATCHMATCH_H
