#ifndef PATCHMATCH_OFFSETMAP_H
#define PATCHMATCH_OFFSETMAP_H

#include <opencv2/imgproc/imgproc.hpp>


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

    bool isFlipped() const { return _flipped; };
    void flip() { _flipped = !_flipped; };

    float get75PercentileDistance() const;

    const int _height, _width;

    /**
     * Some utilities for producing debugging output,
     */
    cv::Mat getDistanceImage() const;
    cv::Mat toColorCodedImage() const;
    double summedDistance() const;

private:
    std::vector<OffsetMapEntry> _data;
    bool _flipped = false;
};

#endif //PATCHMATCH_OFFSETMAP_H
