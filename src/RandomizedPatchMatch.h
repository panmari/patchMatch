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
    cv::Mat _offset_map;
    // Finds number of scales. At minimum scale, both source & target should still be larger than 2 * patch_size in
    // their minimal dimension.
    int findNumberScales(cv::Mat &source, cv::Mat &target, int patch_size) const;

private:
    std::vector<cv::Mat> _source_pyr, _target_pyr, _offset_map_pyr;
    const int _patch_size, _max_search_radius;
    // Minimum size image in pyramid is 2x patchSize of lower dimension (or larger).
    const int _nr_scales;

    // Mainly for debugging, dumps offset map to file.
    void dumpOffsetMapToFile(cv::Mat &offset_map, cv::String filename_modifier) const;
    void initializeWithRandomOffsets(cv::Mat &target_img, cv::Mat &source_img, cv::Mat &offset_map);
    void updateOffsetMapEntryIfBetter(cv::Mat &patch, cv::Point &candidate_offset,
                                      cv::Rect &candiadate_rect, cv::Mat &source_img, cv::Vec3f *offset_map_entry);

};


#endif //PATCHMATCH_RANDOMIZEDPATCHMATCH_H
