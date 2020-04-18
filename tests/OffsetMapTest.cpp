#include "gtest/gtest.h"
#include "../src/OffsetMap.h"

TEST(offset_map_test, flipping_should_work_on_square_image)
{
    OffsetMap test = OffsetMap(101, 101);
    ASSERT_FALSE(test.isFlipped());

    OffsetMapEntry *middle = test.ptr(50, 50);
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
    top_left->distance = 100;

    // Check if it was really written there:
    OffsetMapEntry top_left_copy = test.at(0, 0);
    ASSERT_EQ(100, top_left_copy.distance);

    // Flip offset map, test again.
    test.flip();
    ASSERT_TRUE(test.isFlipped());
    ASSERT_EQ(100, test._width);
    ASSERT_EQ(100, test._height);

    OffsetMapEntry *bottom_right_flipped = test.ptr(99, 99);
    EXPECT_EQ(100, bottom_right_flipped->distance);
}

TEST(offset_map_test, percentile_distance_should_be_computed_correctly)
{
    OffsetMap test = OffsetMap(4, 1);
    test.ptr(0, 0)->distance = 10;
    test.ptr(0, 1)->distance = 40;
    test.ptr(0, 2)->distance = 60;
    test.ptr(0, 3)->distance = 1000;

    float gotten_percentile = test.get75PercentileDistance();
    ASSERT_EQ(60, gotten_percentile);
}
