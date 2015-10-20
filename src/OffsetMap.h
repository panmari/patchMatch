#ifndef PATCHMATCH_OFFSETMAP_H
#define PATCHMATCH_OFFSETMAP_H

#include "opencv2/imgproc/imgproc.hpp"


class OffsetMapEntry {
public:
    cv::Point offset;
    float distance;
    // TODO: add more attributes like rotation, gain, bias etc.
};

class OffsetMap {

public:
    OffsetMap(const int width, const int height);
    OffsetMapEntry at(const int y, const int x) const;
    OffsetMapEntry* ptr(const int y, const int x);
    double summedDistance() const;
    bool isFlipped() const { return _flipped; };
    void flip() { _flipped = !_flipped; };

    const int _height, _width;
private:
    std::vector<OffsetMapEntry> _data;
    bool _flipped;
};

#endif //PATCHMATCH_OFFSETMAP_H
