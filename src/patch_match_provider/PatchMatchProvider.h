//
// Abstract superclass for patch match algorithms.
//

#ifndef PATCHMATCH_PATCHMATCHPROVIDER_H
#define PATCHMATCH_PATCHMATCHPROVIDER_H

#include "opencv2/imgproc/imgproc.hpp"

using cv::Mat;

class PatchMatchProvider {
public:
    virtual Mat match() = 0;
};
#endif //PATCHMATCH_PATCHMATCHPROVIDER_H
