#ifndef PATCHMATCH_UTIL_H
#define PATCHMATCH_UTIL_H

#include "opencv2/imgproc/imgproc.hpp"

namespace pmutil {

    using cv::Mat;
    using cv::Scalar;

    /**
     * Computes the sum of squared differences of the two given matrices/images. Assumes that they have the same size
     * and type. 
     */
    static double ssd(Mat &img, Mat &img2) {
        Mat diff = img - img2;
        Mat squares = diff.mul(diff);
        Scalar ssd_channels = sum(squares);
        double ssd = ssd_channels[0] + ssd_channels[1] + ssd_channels[2];
        return ssd;
    }
}

#endif //PATCHMATCH_UTIL_H
