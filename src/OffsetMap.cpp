#include "OffsetMap.h"

using cv::Mat;
using cv::Size;
using std::sort;
using std::transform;
using std::vector;

namespace {
    float map_to_distance(const OffsetMapEntry &entry) {
        return entry.distance;
    }
}

OffsetMap::OffsetMap(const int width, const int height) : _width(width), _height(height), _data(width * height) { }

OffsetMapEntry OffsetMap::at(const int y, const int x) const {
    return _data[y + x * _height];
}

OffsetMapEntry *OffsetMap::ptr(const int y, const int x) {
    return &_data[y + x * _height];
}

float OffsetMap::get75PercentileDistance() const {
    vector<float> distances(_data.size());
    transform(_data.begin(), _data.end(), distances.begin(), map_to_distance);
    sort(distances.begin(), distances.end());
    int percentile_idx = static_cast<int>((distances.size() - 1) * 3.0 / 4.0);
    return distances[percentile_idx];
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

Mat OffsetMap::toColorCodedImage() const {
    Mat hsv_img;
    hsv_img.create(Size(_width, _height), CV_32F);

    Mat angles = Mat::zeros(hsv_img.size(), CV_32FC1);
    Mat magnitudes = Mat::zeros(hsv_img.size(), CV_32FC1);

    // Produce some nice to look at output by coding angle to best patch as hue, magnitude as saturation.
    for (int x = 0; x < hsv_img.cols; x++) {
        for (int y = 0; y < hsv_img.rows; y++) {
            OffsetMapEntry offset_map_entry = at(y, x);
            float x_offset = offset_map_entry.offset.x;
            float y_offset = offset_map_entry.offset.y;
            float angle = atan2f(x_offset, y_offset);
            if (angle < 0)
                angle += CV_2PI;
            angles.at<float>(y, x) = angle / CV_2PI * 360;
            magnitudes.at<float>(y, x) = sqrt(x_offset*x_offset + y_offset*y_offset);
        }
    }
    normalize(magnitudes, magnitudes, 0, 1, cv::NORM_MINMAX, CV_32FC1, Mat() );
    Mat hsv_array[] = {angles, magnitudes, Mat::ones(hsv_img.size(), CV_32FC1)};
    cv::merge(hsv_array, 3, hsv_img);
    cvtColor(hsv_img, hsv_img, CV_HSV2BGR);
    return hsv_img;
}

