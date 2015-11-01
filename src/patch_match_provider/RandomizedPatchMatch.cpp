#include "RandomizedPatchMatch.h"
#include <opencv2/highgui/highgui.hpp>
#include "../util.h"
#include <iostream>

using cv::addWeighted;
using cv::buildPyramid;
using cv::flip;
using cv::Mat;
using cv::Point;
using cv::Range;
using cv::Rect;
using cv::RNG;
using cv::Scalar;
using cv::Size;
using cv::String;
using cv::Vec3f;
using pmutil::ssd_unsafe;
using pmutil::computeGradientX;
using pmutil::computeGradientY;
using std::max;
using std::shared_ptr;

namespace {
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
}

/**
 * Number times propagation/random_search is executed for every patch per iteration.
 */
constexpr int ITERATIONS_PER_SCALE = 5;
/**
 * If true, does try to improve patches by doing random search. Else, only propagation is used. Default: true.
 */
constexpr bool RANDOM_SEARCH = true;
constexpr bool MERGE_UPSAMPLED_OFFSETS = true;
constexpr float ALPHA = 0.5; // Used to modify random search radius. Higher alpha means more random searches.

RandomizedPatchMatch::RandomizedPatchMatch(const cv::Mat &source, const cv::Mat &target, int patch_size, float lambda) :
        _patch_size(patch_size), _max_search_radius(max(target.cols, target.rows)),
        _nr_scales(findNumberScales(source, target, patch_size)), _lambda(lambda) {
    buildPyramid(source, _source_pyr, _nr_scales);
    for (Mat scaled_source: _source_pyr) {
        _source_rect_pyr.push_back(Rect(Point(0,0), scaled_source.size()));
        Mat gx;
        computeGradientX(scaled_source, gx);
        _source_grad_x_pyr.push_back(gx);
        Mat gy;
        computeGradientY(scaled_source, gy);
        _source_grad_y_pyr.push_back(gy);
    }
    setTargetArea(target);
}

shared_ptr<OffsetMap> RandomizedPatchMatch::match() {
    RNG rng(_target_updated_count);

    // Initialize with dummy offset map that will be deleted at the end of first iteration.
    OffsetMap *previous_scale_offset_map = new OffsetMap(0, 0);
    for (int scale = _nr_scales; scale >= 0; scale--) {
        Mat source = _source_pyr[scale];
        Mat target = _target_pyr[scale];
        const int width = target.cols - _patch_size + 1;
        const int height = target.rows - _patch_size + 1;
        OffsetMap *offset_map = new OffsetMap(width, height);
        unsigned int random_seed = static_cast<unsigned int>(target.rows * target.cols + _target_updated_count);
        initializeWithRandomOffsets(source, target, scale, offset_map, random_seed);

        for (int i = 0; i < ITERATIONS_PER_SCALE; i++) {
            // After half the iterations, merge the lower resolution offset where they're better.
            // This has to be done in an 'even' iteration because of the flipping.
            if (MERGE_UPSAMPLED_OFFSETS && scale != _nr_scales && i == ITERATIONS_PER_SCALE / 2) {
                assert(!offset_map->isFlipped());
                parallel_for_(cv::Range(0, previous_scale_offset_map->_width),
                              ParallelMergeOffsetMaps(*previous_scale_offset_map, 2, _patch_size, scale,
                                                      *this, *offset_map));

                // If we're on full resolution and have a previous solution, try to merge it, too.
                if (scale == 0 && _previous_solution != nullptr) {
                    ParallelMergeOffsetMaps pmom(*_previous_solution, 1, _patch_size,
                                                 scale, *this, *offset_map);
                    Range whole_width(0, _previous_solution->_width);
                    pmom(whole_width);
                    //parallel_for_(whole_width, pmom);
                }
            }

            for (int x = 0; x < offset_map->_width; x++) {
                for (int y = 0; y < offset_map->_height; y++) {
                    OffsetMapEntry *offset_map_entry = offset_map->ptr(y, x);

                    // If image is flipped, we need to get x and y coordinates unflipped for getting the right offset.
                    // TODO: Handle the flipping better now that offset_map is a custom class.
                    int x_unflipped, y_unflipped;
                    if (offset_map->isFlipped()) {
                        x_unflipped = offset_map->_width - 1 - x;
                        y_unflipped = offset_map->_height - 1 - y;
                    } else {
                        x_unflipped = x;
                        y_unflipped = y;
                    }
                    Rect target_patch_rect(x_unflipped, y_unflipped, _patch_size, _patch_size);

                    // Propagate step, try offsets of neighboring entries for this one, apply if better.
                    if (x > 0) {
                        OffsetMapEntry offsetLeft = offset_map->at(y, x - 1);
                        Rect rectLeft(offsetLeft.offset.x + x_unflipped, offsetLeft.offset.y + y_unflipped,
                                      _patch_size, _patch_size);
                        updateOffsetMapEntryIfBetter(target_patch_rect, offsetLeft.offset, rectLeft, scale, offset_map_entry);
                    }
                    if (y > 0) {
                        OffsetMapEntry offsetUp = offset_map->at(y - 1, x);
                        Rect rectUp(offsetUp.offset.x + x_unflipped, offsetUp.offset.y + y_unflipped,
                                    _patch_size, _patch_size);
                        updateOffsetMapEntryIfBetter(target_patch_rect, offsetUp.offset, rectUp, scale, offset_map_entry);
                    }

                    // Random search step, try out various locations all over the image that could be better.
                    if (RANDOM_SEARCH) {
                        Point current_offset = offset_map_entry->offset;
                        float current_search_radius = _max_search_radius;
                        while (current_search_radius > 1) {
                            Point random_point = Point(cvRound(rng.uniform(-1.f, 1.f) * current_search_radius),
                                                       cvRound(rng.uniform(-1.f, 1.f) * current_search_radius));
                            Point random_offset = current_offset + random_point;
                            Rect random_rect(x_unflipped + random_offset.x,
                                             y_unflipped + random_offset.y, _patch_size, _patch_size);

                            updateOffsetMapEntryIfBetter(target_patch_rect, random_offset, random_rect, scale,
                                                         offset_map_entry);

                            current_search_radius *= ALPHA;
                        }
                    }
                }
            }
            // Every second iteration, we go the other way round (start at bottom, propagate from right and down).
            // This effect can be achieved by flipping the matrix after every iteration.
            offset_map->flip();
        }
        if (offset_map->isFlipped()) {
            // Correct orientation if we're still in flipped state.
            offset_map->flip();
        }
        delete previous_scale_offset_map;
        previous_scale_offset_map = offset_map;
    }
    _previous_solution = shared_ptr<OffsetMap>(previous_scale_offset_map);
    return _previous_solution;
}

void RandomizedPatchMatch::updateOffsetMapEntryIfBetter(const Rect &target_patch_rect, const Point &candidate_offset,
                                                        const Rect &candidate_rect, const int scale,
                                                        OffsetMapEntry *offset_map_entry) const {
    // Check if it's fully inside, only try to update then.
    if ((_source_rect_pyr[scale] & candidate_rect) == candidate_rect) {
        float previous_distance = offset_map_entry->distance;
		float ssd_value = patchDistance(candidate_rect, target_patch_rect, scale, previous_distance);
        if (ssd_value < previous_distance) {
            offset_map_entry->offset = candidate_offset;
            offset_map_entry->distance = ssd_value;
        }
    }
}

void RandomizedPatchMatch::setTargetArea(const cv::Mat &new_target_area) {
    _target_updated_count++;
    buildPyramid(new_target_area, _target_pyr, _nr_scales);
    _target_grad_x_pyr.resize(0);
    _target_grad_y_pyr.resize(0);
    for (Mat scaled_target: _target_pyr) {
        Mat gx;
        computeGradientX(scaled_target, gx);
        _target_grad_x_pyr.push_back(gx);
        Mat gy;
        computeGradientY(scaled_target, gy);
        _target_grad_y_pyr.push_back(gy);
    }
}


void RandomizedPatchMatch::initializeWithRandomOffsets(const Mat &source_img, const Mat &target_img, const int scale,
                                                       OffsetMap *offset_map, unsigned int random_seed) const {
    // Seed random generator to have reproducable results.
    srand(random_seed);
    for (int x = 0; x < offset_map->_width; x++) {
        for (int y = 0; y < offset_map->_height; y++) {
            // Choose offset carefully, so resulting point (when added to current coordinate), is not outside image.
            int randomX = (rand() % (source_img.cols - _patch_size)) - x;
            int randomY = (rand() % (source_img.rows - _patch_size)) - y;

            Rect current_patch_rect(x, y, _patch_size, _patch_size);
            Rect random_rect = Rect(x + randomX, y + randomY, _patch_size, _patch_size);
			float inital_dist = patchDistance(random_rect, current_patch_rect, scale);
            auto entry = offset_map->ptr(y, x);
            entry->offset = Point(randomX, randomY);
            entry->distance = inital_dist;
        }
    }
}

int RandomizedPatchMatch::findNumberScales(const Mat &source, const Mat &target, int patch_size) const {
    double min_dimension = std::min(std::min(std::min(source.cols, source.rows), target.cols), target.rows);
    return cvFloor(log2( min_dimension / patch_size));
}

float RandomizedPatchMatch::patchDistance(const Rect &source_rect, const Rect &target_rect, const int scale,
                                          const float previous_dist) const {
    Mat source_patch = _source_pyr[scale](source_rect);
    Mat target_patch = _target_pyr[scale](target_rect);
    double ssd = ssd_unsafe(source_patch, target_patch, previous_dist);

    // Computation can be canceled early if distance is higher than previous distance (or gradients are not used).
    if (ssd > previous_dist || _lambda == 0)
        return static_cast<float>(ssd);

    Mat source_grad_x_patch = _source_grad_x_pyr[scale](source_rect);
    Mat target_grad_x_patch = _target_grad_x_pyr[scale](target_rect);
    double ssd_grad_x = ssd_unsafe(source_grad_x_patch, target_grad_x_patch);
    ssd += ssd_grad_x * _lambda;

    if (ssd > previous_dist)
        return static_cast<float>(ssd);

    Mat source_grad_y_patch = _source_grad_y_pyr[scale](source_rect);
    Mat target_grad_y_patch = _target_grad_y_pyr[scale](target_rect);
    double ssd_grad_y = ssd_unsafe(source_grad_y_patch, target_grad_y_patch);
    ssd += ssd_grad_y * _lambda;

    return static_cast<float>(ssd);
}