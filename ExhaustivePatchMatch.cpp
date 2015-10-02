//
// Created by moser on 02.10.15.
//

#include "ExhaustivePatchMatch.h"
#include <boost/progress.hpp>

using namespace std;
using namespace cv::cuda;

ExhaustivePatchMatch::ExhaustivePatchMatch(Mat &img, Mat &img2) {
    _img.upload(img);
    _img2.upload(img2);
    _cuda_matcher = createTemplateMatching(CV_8UC3, CV_TM_SQDIFF);
    _temp.create(_img.rows, _img.cols, CV_32FC1);
}

Mat ExhaustivePatchMatch::match(int patchSize) {
    // Create the result matrix
    Mat minDistImg;
    minDistImg.create(_img.rows, _img.cols, CV_32FC1);

    static int matched_pixels = (_img.cols - 2 * patchSize) * (_img.rows - 2 * patchSize);
    boost::progress_display show_progress(matched_pixels);
    boost::timer timer;
    for (int x = patchSize; x < _img.cols - patchSize; x++) {
        for (int y = patchSize; y < _img.rows - patchSize; y++) {
            Rect rect(x, y, patchSize, patchSize);
            GpuMat patch = _img2(rect);
            double minVal; Point minLoc;
            matchSinglePatch(patch, &minVal, &minLoc);
            minDistImg.at<float>(Point(x,y)) = (float)minVal;
            ++show_progress;
        }
    }
    cout << timer.elapsed() << endl;
    return minDistImg;
}

void ExhaustivePatchMatch::matchSinglePatch(GpuMat &patch, double *minVal, Point *minLoc) {
    // Do the Matching
    _cuda_matcher->match(_img, patch, _temp);
    // Localizing the best match with minMaxLoc
    cuda::minMaxLoc(_temp, minVal, nullptr, minLoc, nullptr);
    return;
}
