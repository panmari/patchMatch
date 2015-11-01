#ifndef PATCHMATCH_PARALLELMERGEOFFSETMAPS_H
#define PATCHMATCH_PARALLELMERGEOFFSETMAPS_H

#include <opencv2/imgproc/imgproc.hpp>
#include "../OffsetMap.h"
#include "RandomizedPatchMatch.h"

using cv::Rect;
using cv::Point;
using cv::Size;

class ParallelMergeOffsetMaps : public cv::ParallelLoopBody {
private:
    const OffsetMap &_other_offset_map;
    const int _scale_difference, _patch_size, _current_scale;
    OffsetMap &_offset_map;
    RandomizedPatchMatch &_rmp;

public:
    ParallelMergeOffsetMaps(const OffsetMap &other_offset_map, const int scale_difference,
                            const int patch_size, const int current_scale,
                            RandomizedPatchMatch &rmp, OffsetMap &offset_map)
            : _other_offset_map(other_offset_map), _current_scale(current_scale),
              _scale_difference(scale_difference), _patch_size(patch_size),
              _rmp(rmp), _offset_map(offset_map) { }

    virtual void operator()(const cv::Range &r) const {
        Size sz(_patch_size, _patch_size);
        for (int x = r.start; x < r.end; x++) {
            for (int y = 0; y < _other_offset_map._height; y++) {
                Point other_offset = _other_offset_map.at(y, x).offset * _scale_difference;
                Point offset_map_at(x * _scale_difference, y * _scale_difference);
                Rect target_patch_rect(offset_map_at, sz);
                Rect other_rect(other_offset, sz);
                OffsetMapEntry *current_offset = _offset_map.ptr(y * _scale_difference,
                                                                 x * _scale_difference);
                _rmp.updateOffsetMapEntryIfBetter(target_patch_rect, other_offset,
                                                  other_rect, _current_scale, current_offset);
            }
        }
    }
};

#endif //PATCHMATCH_PARALLELMERGEOFFSETMAPS_H
