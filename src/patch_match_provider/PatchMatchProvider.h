//
// Abstract superclass for patch match algorithms.
//

#ifndef PATCHMATCH_PATCHMATCHPROVIDER_H
#define PATCHMATCH_PATCHMATCHPROVIDER_H

using cv::Mat;

class PatchMatchProvider {
public:
    virtual Mat match() = 0;
};
#endif //PATCHMATCH_PATCHMATCHPROVIDER_H
