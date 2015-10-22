#include "OffsetMap.h"

using cv::Mat;
using cv::Size;

OffsetMap::OffsetMap(const int width, const int height) : _width(width), _height(height), _data(width * height) { }

OffsetMapEntry OffsetMap::at(const int y, const int x) const {
    if (_flipped) {
        const int x_flipped = _width - x - 1;
        const int y_flipped = _height - y - 1;
        return _data[y_flipped + x_flipped * _height];
    } else {
        return _data[y + x * _height];
    }
}

OffsetMapEntry *OffsetMap::ptr(const int y, const int x) {
    if (_flipped) {
        const int x_flipped = _width - x - 1;
        const int y_flipped = _height - y - 1;
        return &_data[y_flipped + x_flipped * _height];
    } else {
        return &_data[y + x * _height];
    }
}

double OffsetMap::summedDistance() const {
    double sum = 0;
    for(auto &entry: _data) {
        sum += entry.distance;
    }
    return sum;
}

Mat OffsetMap::getDistanceImage() const {
    Mat dist_image;
    dist_image.create(Size(_width, _height), CV_32F);
    for (int x = 0; x < dist_image.cols; x++) {
        for (int y = 0; y < dist_image.rows; y++) {
            dist_image.at<float>(y, x) = at(y, x).distance;
        }
    }
    return dist_image;
}
