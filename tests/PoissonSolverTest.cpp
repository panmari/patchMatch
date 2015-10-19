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

TEST(poisson_solver_test, noisy_synthetic_image_given) {
    Size full_size(500, 500);
    Mat img(full_size, CV_32FC3);
    randu(img, 0.0, 1.0);

    Mat grad_x, grad_y;
    computeGradientX(img, grad_x);
    computeGradientY(img, grad_y);

    Mat noise(full_size, CV_32FC3);
    randn(noise, 0, 0.01);

    // Add some noise to image.
    Mat img_noisy = img + noise;
    // Use noisy image with good gradients to reconstruct.
    PoissonSolver ps(img_noisy, grad_x, grad_y);

    Mat result;
    ps.solve(result);
    imwrite("poisson_original.exr", img);
    imwrite("poisson_solved.exr", result);
    // Error here is quite high.
    ASSERT_NEAR(ssd(result, img), 0, 2);
}

TEST(poisson_solver_test, noisy_natural_image_given) {
    Mat img = imread("test_images/unitobler.jpg");
    // Make smaller, bring to range [0, 1]
    const float resize_factor = 0.25f;
    resize(img, img, Size(), resize_factor, resize_factor);
    img.convertTo(img, CV_32FC3, 1 / 255.f);

    Mat grad_x, grad_y;
    computeGradientX(img, grad_x);
    computeGradientY(img, grad_y);

    Mat noise(img.size(), CV_32FC3);
    randn(noise, 0, 0.01);

    // Add some noise to image.
    Mat img_noisy = img + noise;
    // Use noisy image with good gradients to reconstruct.
    PoissonSolver ps(img_noisy, grad_x, grad_y);

    Mat result;
    ps.solve(result);
    imwrite("poisson_original.exr", img);
    imwrite("poisson_solved.exr", result);
    ASSERT_NEAR(ssd(result, img), 0, 1);
}