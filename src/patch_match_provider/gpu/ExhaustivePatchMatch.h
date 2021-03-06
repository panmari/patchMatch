#ifndef PATCHMATCH_EXHAUSTIVEPATCHMATCH_H
#define PATCHMATCH_EXHAUSTIVEPATCHMATCH_H

#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "../PatchMatchProvider.h"

class ExhaustivePatchMatch : public PatchMatchProvider {

public:
	ExhaustivePatchMatch(cv::Mat &source, cv::Mat &target, int patch_size, bool show_progress_bar = false);
    std::shared_ptr<OffsetMap> match();

private:
    bool _show_progress_bar;
	int _patch_size;
    cv::cuda::GpuMat _source, _target, _temp;
    cv::Ptr<cv::cuda::TemplateMatching> _cuda_matcher;

    void matchSinglePatch(cv::cuda::GpuMat &patch, double *minVal, cv::Point *minLoc);
};


#endif //PATCHMATCH_EXHAUSTIVEPATCHMATCH_H
