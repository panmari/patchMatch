//
// Created by moser on 02.10.15.
//

#include "ExhaustivePatchMatch.h"
#include <boost/progress.hpp>

using cv::cuda::createTemplateMatching;
using cv::cuda::GpuMat;
using cv::Mat;
using cv::Point;
using cv::Rect;
using cv::Scalar;
using cv::Vec3f;

ExhaustivePatchMatch::ExhaustivePatchMatch(Mat &img, Mat &img2) {
    _img.upload(img);
    _img2.upload(img2);
    _cuda_matcher = createTemplateMatching(CV_32FC3, CV_TM_SQDIFF);
    _temp.create(_img.rows, _img.cols, CV_32FC1);
}

Mat ExhaustivePatchMatch::match(int patchSize) {
    // Create the result matrix
    Mat minDistImg;
    minDistImg.create(_img.rows, _img.cols, CV_32FC3);

    const int matched_pixels = (_img.cols - 2 * patchSize) * (_img.rows - 2 * patchSize);
    boost::progress_display show_progress(matched_pixels);
    boost::timer timer;
    for (int x = patchSize; x < _img.cols - patchSize; x++) {
        for (int y = patchSize; y < _img.rows - patchSize; y++) {
            Rect rect(x, y, patchSize, patchSize);
            GpuMat patch = _img2(rect);
            double minVal; Point minLoc;
            matchSinglePatch(patch, &minVal, &minLoc);
            minDistImg.at<Vec3f>(y, x) = Vec3f(minLoc.x, minLoc.y, minVal);
            ++show_progress;
        }
    }
    std::cout << timer.elapsed() << std::endl;
    Scalar squared_diff = sum(minDistImg);
    std::cout << squared_diff[2] << std::endl;
    return minDistImg;
}

void ExhaustivePatchMatch::matchSinglePatch(GpuMat &patch, double *minVal, Point *minLoc) {
    // Do the Matching
    _cuda_matcher->match(_img, patch, _temp);
    // Localizing the best match with minMaxLoc
    cv::cuda::minMaxLoc(_temp, minVal, nullptr, minLoc, nullptr);
    return;
}