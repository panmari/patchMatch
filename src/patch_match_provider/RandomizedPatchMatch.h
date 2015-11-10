// Implements randomized patch match, as described in
// PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing
// by Barnes et al.
#ifndef PATCHMATCH_RANDOMIZEDPATCHMATCH_H
#define PATCHMATCH_RANDOMIZEDPATCHMATCH_H

#include <opencv2/imgproc/imgproc.hpp>
#include "PatchMatchProvider.h"
#include "../OffsetMap.h"

class RandomizedPatchMatch : public PatchMatchProvider {

public:
    /**
     * Constructs all things necessary to execute randomized patch match on the given source image. The target image
     * has to be set via setTargetArea before calling the match() which does the actual patch matching.
     * Additional to translation, also rotated versions of the image will be inspected.
     *
     * @param min_rotation the minimal rotation inspected
     * @param max_rotation the maximal rotation inspected
     * @param rotation_step decides the number of rotations considered. Will construct rotated versions of the image
     * until min_rotation + i*rotation_step > max_rotation.
     * You are advised to choose min_rotation and rotation_step so that the rotation by 0 degrees is also included.
     */
    RandomizedPatchMatch(const cv::Mat &source, const cv::Size &target_size, int patch_size,
                         float lambda = 0.5f, float min_rotation = -10, float max_rotation = 10,
                         float rotation_step = 5);
    std::shared_ptr<OffsetMap> match() override;

    /* Finds number of scales. At minimum scale, both source & target should still be larger than 2 * patch_size in
     * their minimal dimension.
     */
    int findNumberScales(const cv::Size &source_size, const cv::Size &target_size, int patch_size) const;

    void setTargetArea(const cv::Mat &new_target_area);
    const std::vector<cv::Mat> getSourcesRotated() const { return _source_rotations_pyr[0]; };
    const cv::Mat &getSourceGradientX() const { return _source_grad_x_pyr[0]; };
    const cv::Mat &getSourceGradientY() const { return _source_grad_y_pyr[0]; };

    /**
    * Updates 'offset_map_entry' with the given 'candidate_offset' if the patch corresponding to 'candidate_rect' on
    * 'source_img' is a better match than for the given 'patch'.
    */
    void updateOffsetMapEntryIfBetter(const cv::Rect &target_patch_rect, const OffsetMapEntry &candidate,
                                      const int scale, OffsetMapEntry *offset_map_entry) const;


private:
    std::vector<cv::Mat> _source_pyr, _target_pyr;

    /**
     * Gradients
     */
    std::vector<cv::Mat> _source_grad_x_pyr, _source_grad_y_pyr, _target_grad_x_pyr, _target_grad_y_pyr;
    std::vector<std::vector<cv::Mat>> _source_rotations_pyr;
    std::vector<cv::Rect> _source_rect_pyr;
    const int _patch_size, _max_search_radius;
    // Minimum size image in pyramid is 2x patchSize of lower dimension (or larger).
    const int _nr_scales;

    /**
     * Weight of gradient in distance measure, should be in [0, 1]. Default is 0.5.
     */
    const float _lambda;

    /**
     * Used for initializing RNG independently over multiple EM runs.
     */
    int _target_updated_count = 0;
    std::shared_ptr<OffsetMap> _previous_solution = nullptr;

    /* Mainly for debugging, dumps offset map to file. */
    void dumpOffsetMapToFile(cv::Mat &offset_map, cv::String filename_modifier) const;

    /*
     * Every entry at offset_map is set to a random & valid (i. e. patch it's pointing to is inside image) offset.
     * Also the corresponding SSD is computed.
     */
    void initializeWithRandomOffsets(const cv::Mat &target_img, const cv::Mat &source_img, const int scale,
                                     OffsetMap *offset_map, unsigned int random_seed = 42) const;

    /**
     * Computes the distance of two patches. Patches have to be the same size on both images.
     * Uses internally the parameter '_lambda' to weight distance of gradients.
     * @param source_rect the position of the patch on the source image.
     * @param target_rect the position of the patch on the target image.
     * @param scale the scale that the rectangles reference to.
     */
    float patchDistance(const cv::Rect &source_rect, const cv::Rect &target_rect, const int scale,
                        const float previous_dist = INFINITY) const;
};

#endif //PATCHMATCH_RANDOMIZEDPATCHMATCH_H
