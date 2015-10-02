//
// Created by moser on 02.10.15.
//

#include "RandomizedPatchMatch.h"
#include "opencv2/highgui/highgui.hpp"

using cv::Vec3f;
using cv::imshow;

RandomizedPatchMatch::RandomizedPatchMatch(cv::Mat &img, cv::Mat &img2) : _img(img), _img2(img2) {
    _offset_map.create(_img.rows, _img.cols, CV_32FC3);
    for (int x = 0; x < _offset_map.cols; x++) {
        for (int y = 0; y < _offset_map.rows; y++) {
            int randomX = (rand() % _offset_map.cols);
            int randomY = (rand() % _offset_map.rows);
            _offset_map.at<Vec3f>(y, x) = Vec3f(randomX, randomY, 0);
        }
    }
    imshow("tst", _offset_map);
}
