#include "OffsetMap.h"

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
