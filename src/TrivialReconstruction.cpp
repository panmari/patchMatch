#include "TrivialReconstruction.h"
#include <iostream>

using cv::Mat;
using cv::Rect;
using cv::Vec3f;
using std::shared_ptr;

TrivialReconstruction::TrivialReconstruction(const shared_ptr<OffsetMap>offset_map, const Mat &source_img, const int patch_size) :
        _offset_map(offset_map), _source_img(source_img), _patch_size(patch_size) { }

Mat TrivialReconstruction::reconstruct() const {
    Mat reconstructed = Mat::zeros(_offset_map->_height + _patch_size, _offset_map->_width + _patch_size, CV_32FC3);
    for (int x = 0; x < _offset_map->_width; x++) {
        for (int y = 0; y < _offset_map->_height; y++) {
            OffsetMapEntry offset_map_entry = _offset_map->at(y, x);
            int match_x = x + offset_map_entry.offset.x;
            int match_y = y + offset_map_entry.offset.y;

            if (match_x < 0 || match_x >= _source_img.cols || match_y < 0 || match_y >= _source_img.rows) {
                std::cout << "Not inside image: (" << match_x << "," << match_y << ")" << std::endl;
            }

            Rect matching_patch_rect(match_x, match_y, _patch_size, _patch_size);
            Mat matching_patch = _source_img(matching_patch_rect);

            Rect current_patch_rect(x, y, _patch_size, _patch_size);

            // Simple assignment doesn't work here, need to copy explicitly.
            matching_patch.copyTo(reconstructed(current_patch_rect));
        }
    }
    return reconstructed;
}