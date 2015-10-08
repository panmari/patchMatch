//
// Created by panmari on 02.10.15.
//

#ifndef PATCHMATCH_RANDOMIZEDPATCHMATCH_H
#define PATCHMATCH_RANDOMIZEDPATCHMATCH_H

#include "opencv2/imgproc/imgproc.hpp"

class RandomizedPatchMatch {

public:
    RandomizedPatchMatch(cv::Mat &img, cv::Mat &img2, int patchSize);
    cv::Mat match();
    cv::Mat _offset_map;

private:
    std::vector<cv::Mat> _img_pyr, _img2_pyr;
    const int _patchSize, _max_sarch_radius;
    // Minimum size image in pyramid is 2x patchSize of lower dimension (or larger).
    const int _nr_scales;

    // Mainly for debugging, dumps offset map to file.
    void dumpOffsetMapToFile(cv::Mat &offset_map, cv::String filename_modifier) const;
    double ssd(cv::Mat &patch, cv::Mat &patch2) const;
    void initializeWithRandomOffsets(cv::Mat &img, cv::Mat &img2, cv::Mat &offset_map);
    void updateOffsetMapEntryIfBetter(cv::Mat &patch, cv::Point &candidate_offset,
                                      cv::Rect &candiadate_rect, cv::Mat &other_img, cv::Vec3f *offset_map_entry);
};


#endif //PATCHMATCH_RANDOMIZEDPATCHMATCH_H
