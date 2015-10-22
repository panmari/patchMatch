#include "opencv2/ts.hpp"
#include "../src/OffsetMap.h"

TEST(offset_map_test, flipping_should_work_on_square_image)
{
    OffsetMap test = OffsetMap(101, 101);
    ASSERT_FALSE(test.isFlipped());

    OffsetMapEntry *middle = test.ptr(50, 50);
    ASSERT_EQ(0, middle->distance);
    middle->distance = 100;

    // Check if it was really written there:
    OffsetMapEntry middle_copy = test.at(50, 50);
    ASSERT_EQ(100, middle_copy.distance);

    // Flip offset map, test again.
    test.flip();
    ASSERT_TRUE(test.isFlipped());

    OffsetMapEntry middle_copy_flipped = test.at(50, 50);
    ASSERT_EQ(100, middle_copy_flipped.distance);
}

TEST(offset_map_test, flipping_should_work_on_square_image_for_top_left)
{
    OffsetMap test = OffsetMap(100, 100);
    ASSERT_FALSE(test.isFlipped());

    OffsetMapEntry *top_left = test.ptr(0, 0);
    ASSERT_EQ(0, top_left->distance);
    top_left->distance = 100;

    // Check if it was really written there:
    OffsetMapEntry top_left_copy = test.at(0, 0);
    ASSERT_EQ(100, top_left_copy.distance);

    // Flip offset map, test again.
    test.flip();
    ASSERT_TRUE(test.isFlipped());

    OffsetMapEntry *top_left_flipped = test.ptr(0, 0);
    EXPECT_NE(top_left, top_left_flipped);
    EXPECT_EQ(0, top_left_flipped->distance);

    OffsetMapEntry *bottom_right_flipped = test.ptr(99, 99);
    EXPECT_EQ(top_left, bottom_right_flipped);

    EXPECT_EQ(100, bottom_right_flipped->distance);
}
