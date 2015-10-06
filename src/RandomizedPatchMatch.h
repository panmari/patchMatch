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

private:
    const cv::Mat _img, _img2;
    cv::Mat _offset_map;
    const cv::Rect _rect_full_img2;
    const int _patchSize, _max_sarch_radius;

    // Mainly for debugging, dumps offset map to file.
    void dumpOffsetMapToFile(cv::String filename_modifier) const;
    double ssd(cv::Mat &patch, cv::Mat &patch2) const;
    void initializeOffsets(int patchSize);
    void updateOffsetMapEntryIfBetter(cv::Mat &patch, cv::Point &candidate_offset,
                                      cv::Rect &candiadate_rect, cv::Vec3f *offset_map_entry);
};


#endif //PATCHMATCH_RANDOMIZEDPATCHMATCH_H
