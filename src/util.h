#ifndef PATCHMATCH_UTIL_H
#define PATCHMATCH_UTIL_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace pmutil {

    using cv::imwrite;
    using cv::Mat;
    using cv::Scalar;
    using cv::Size;
    using cv::String;

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

    /**
     * Convert images to lab retrieved from imread.
     * L*a*b has the following ranges for each channel:
     * L: [0, 100]
     * a*: [-170, 100]
     * b*: [-100, 150]
     */
    static void convert_for_computation(Mat &img, float resize_factor) {
        if (resize_factor != 1.f) {
            resize(img, img, Size(), resize_factor, resize_factor);
        }
        img.convertTo(img, CV_32FC3, 1 / 255.f);
        cvtColor(img, img, CV_BGR2Lab);
    }

    /**
     * Writes the given img to a file with the given filename, converting it first from lab to bgr
     */
    static void imwrite_lab( String filename, Mat &img) {
        Mat bgr;
        cvtColor(img, bgr, CV_Lab2BGR);
        imwrite(filename, bgr);
    }
}

#endif //PATCHMATCH_UTIL_H
