#ifndef PATCHMATCH_EXHAUSTIVEPATCHMATCH_H
#define PATCHMATCH_EXHAUSTIVEPATCHMATCH_H

#include "opencv2/imgproc/imgproc.hpp"
#include "../PatchMatchProvider.h"

class ExhaustivePatchMatch : public PatchMatchProvider {

public:
	ExhaustivePatchMatch(const cv::Mat &source, const cv::Mat &target, int patch_size, bool show_progress_bar = false);
    void match(OffsetMap *offset_map) override;

private:
    bool _show_progress_bar;
	int _patch_size;
    const cv::Mat _source, _target;
    cv::Mat _temp;

    void matchSinglePatch(const cv::Mat &patch, double *minVal, cv::Point *minLoc) const;
};


#endif //PATCHMATCH_EXHAUSTIVEPATCHMATCH_H
