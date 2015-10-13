#ifndef PATCHMATCH_UTIL_H
#define PATCHMATCH_UTIL_H

#include "boost/format.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace pmutil {

    using boost::format;
    using cv::imwrite;
    using cv::Mat;
    using cv::Rect;
    using cv::Scalar;
    using cv::Size;
    using cv::String;
    using cv::Vec3f;

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

    /**
     * Dumps all nearest patches for visual inspection
     */
    static void dumpNearestPatches(Mat &offset_map, Mat &source, int patch_size, String filename_prefix) {
        for (int x = 0; x < offset_map.cols; x++) {
            for(int y = 0; y < offset_map.rows; y++) {
                Vec3f offset = offset_map.at<Vec3f>(y, x);
                Rect nearest_patch_rect(x + offset[0], y + offset[1], patch_size, patch_size);
                Mat nearest_patch = source(nearest_patch_rect);
                imwrite(str(format("patch_%s_x_%03d_y_%03d.exr") % filename_prefix % x % y), nearest_patch);
            }
        }
    }
}

#endif //PATCHMATCH_UTIL_H
