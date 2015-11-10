#ifndef PATCHMATCH_OFFSETMAP_H
#define PATCHMATCH_OFFSETMAP_H

#include <opencv2/imgproc/imgproc.hpp>


class OffsetMapEntry {
public:
    cv::Point offset;
    float distance;
    unsigned int rotation_idx;
    // TODO: add more attributes like rotation, gain, bias etc.

    /**
     * May return an empty matrix if the patch to be extracted is not inside the image.
     */
    const cv::Mat extractFrom(const std::vector<cv::Mat> &srcs, const int x, const int y,
                              const int patch_size, const int scale_change = 1) const {
        const cv::Mat &src = srcs[rotation_idx];
        cv::Rect roi = cv::Rect((offset.x + x) * scale_change, (offset.y + y) * scale_change,
                                patch_size * scale_change, patch_size * scale_change);
        if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= src.cols && roi.y + roi.width <= src.rows)
            return src(roi);
        else
            return cv::Mat();
    }

    void merge(const OffsetMapEntry &other, float d) {
        this->offset = other.offset;
        this->rotation_idx = other.rotation_idx;
        this->distance = d;
    }
};

class OffsetMap {

public:
    OffsetMap(const int width, const int height);
    OffsetMapEntry at(const int y, const int x) const;
    OffsetMapEntry* ptr(const int y, const int x);

    bool isFlipped() const { return _flipped; };
    void flip() {
        std::reverse(_data.begin(), _data.end());
        _flipped = !_flipped;
    };

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
