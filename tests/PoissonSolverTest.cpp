#include "opencv2/ts.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../src/util.h"
#include "../src/PoissonSolver.h"

using namespace std;
using namespace cv;
using namespace pmutil;

TEST(poisson_solver_test, exact_gradients_given) {
    Size full_size(1000, 1000);
    Mat img(full_size, CV_32FC3);
    randu(img, 0.0, 1.0);

    Mat grad_x, grad_y;
    computeGradientX(img, grad_x);
    computeGradientY(img, grad_y);

    PoissonSolver ps(img, grad_x, grad_y);

    Mat result;
    ps.solve(result);
    ASSERT_NEAR(ssd(result, img), 0, 1);
}

// TODO: Investigate why SSD is rather high in this reconstruction (especially at edges).
TEST(poisson_solver_test, noisy_synthetic_image_given) {
    Size full_size(500, 500);
    Mat img(full_size, CV_32FC3);

    img.setTo(Scalar(1, 1, 1));
    img(Rect(100, 100, 10, 10)) = Scalar(1, 0, 0);
    img(Rect(300, 300, 50, 50)) = Scalar(0, 1, 0);
    img(Rect(200, 300, 10, 50)) = Scalar(0, 0, 1);

    Mat grad_x, grad_y;
    computeGradientX(img, grad_x);
    computeGradientY(img, grad_y);

    Mat noise(full_size, CV_32FC3);
    randn(noise, 0, 0.05);

    // Add some noise to image.
    Mat img_noisy = img + noise;
    // Use noisy image with good gradients to reconstruct.
    PoissonSolver ps(img_noisy, grad_x, grad_y);

    Mat result;
    ps.solve(result);
    imwrite("poisson_original.exr", img);
    imwrite("poisson_solved.exr", result);

    int margin = 5;
    Rect crop_rect(margin, margin, full_size.width - 2 * margin, full_size.height - 2 * margin);
    auto gotten_ssd = ssd(result(crop_rect), img(crop_rect));
    // Should have at least improved image.
    ASSERT_LT(gotten_ssd, ssd(img_noisy(crop_rect), img(crop_rect)));
    auto expected_ssd = 0;
    // Error here is quite high.
    ASSERT_NEAR(gotten_ssd, expected_ssd, 8);
}

TEST(poisson_solver_test, noisy_natural_image_given) {
    Mat img = imread("test_images/unitobler.jpg");
    // Make smaller, bring to range [0, 1]
    const float resize_factor = 0.25f;
    resize(img, img, Size(), resize_factor, resize_factor);
    img.convertTo(img, CV_32FC3, 1 / 255.f);
    Size full_size = img.size();

    Mat grad_x, grad_y;
    computeGradientX(img, grad_x);
    computeGradientY(img, grad_y);

    Mat noise(img.size(), CV_32FC3);
    randn(noise, 0, 0.05);

    // Add some noise to image.
    Mat img_noisy = img + noise;
    // Use noisy image with good gradients to reconstruct.
    PoissonSolver ps(img_noisy, grad_x, grad_y);

    Mat result;
    ps.solve(result);
    imwrite("poisson_original.exr", img);
    imwrite("poisson_solved.exr", result);

    int margin = 0;
    Rect crop_rect(margin, margin, full_size.width - 2 * margin, full_size.height - 2 * margin);
    auto gotten_ssd = ssd(result(crop_rect), img(crop_rect));
    // Should have at least improved image.
    ASSERT_LT(gotten_ssd, ssd(img_noisy(crop_rect), img(crop_rect)));
    auto expected_ssd = 0;
    // Error here is quite high.
    ASSERT_NEAR(gotten_ssd, expected_ssd, 16);
}