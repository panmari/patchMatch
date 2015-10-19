// Taken from
// https://github.com/Itseez/opencv/blob/ddf82d0b154873510802ef75c53e628cd7b2cb13/modules/photo/src/seamless_cloning_impl.cpp
#ifndef PATCHMATCH_POISSON_H
#define PATCHMATCH_POISSON_H

#include "opencv2/imgproc/imgproc.hpp"

namespace poisson {

    using cv::filter2D;
    using cv::Mat;
    using cv::Point;
    using cv::Rect;
    using cv::Scalar;

    namespace {
        // Somehow computes dft of something between src & dst.
        void dst(const Mat &src, Mat &dest, bool invert = false) {
            Mat temp = Mat::zeros(src.rows, 2 * src.cols + 2, CV_32F);

            int flag = invert ? cv::DFT_ROWS + cv::DFT_SCALE + cv::DFT_INVERSE : cv::DFT_ROWS;

            src.copyTo(temp(Rect(1, 0, src.cols, src.rows)));

            for (int j = 0; j < src.rows; ++j) {
                float *tempLinePtr = temp.ptr<float>(j);
                const float *srcLinePtr = src.ptr<float>(j);
                for (int i = 0; i < src.cols; ++i) {
                    // how does src.cols + 2 + i make sense?
                    // We'll be on the next row most probably, so temp should be continuous.
                    tempLinePtr[src.cols + 2 + i] = -srcLinePtr[src.cols - 1 - i];
                }
            }

            Mat planes[] = {temp, Mat::zeros(temp.size(), CV_32F)};
            Mat complex;

            merge(planes, 2, complex);
            dft(complex, complex, flag);
            split(complex, planes);
            temp = Mat::zeros(src.cols, 2 * src.rows + 2, CV_32F);

            for (int j = 0; j < src.cols; ++j) {
                float *tempLinePtr = temp.ptr<float>(j);
                for (int i = 0; i < src.rows; ++i) {
                    float val = planes[1].ptr<float>(i)[j + 1];
                    tempLinePtr[i + 1] = val;
                    tempLinePtr[temp.cols - 1 - i] = -val;
                }
            }

            Mat planes2[] = {temp, Mat::zeros(temp.size(), CV_32F)};

            merge(planes2, 2, complex);
            dft(complex, complex, flag);
            split(complex, planes2);

            temp = planes2[1].t();
            dest = Mat::zeros(src.size(), CV_32F);
            temp(Rect(0, 1, src.cols, src.rows)).copyTo(dest);
        }

        void idst(const Mat &src, Mat &dest) {
            dst(src, dest, true);
        }
    }

    void solve(const Mat &img, Mat &mod_diff, Mat &result) {
        const int w = img.cols;
        const int h = img.rows;
        std::vector<float> filter_X, filter_Y;

        Mat res;
        dst(mod_diff, res);

        for (int j = 0; j < h - 2; j++) {
            float *resLinePtr = res.ptr<float>(j);
            for (int i = 0; i < w - 2; i++) {
                resLinePtr[i] /= (filter_X[i] + filter_Y[j] - 4);
            }
        }

        idst(res, mod_diff);

        uchar *resLinePtr = result.ptr<uchar>(0);
        const uchar *imgLinePtr = img.ptr<uchar>(0);
        const float *interpLinePtr = NULL;

        //first col
        for (int i = 0; i < w; ++i)
            result.ptr<uchar>(0)[i] = img.ptr<uchar>(0)[i];

        for (int j = 1; j < h - 1; ++j) {
            resLinePtr = result.ptr<uchar>(j);
            imgLinePtr = img.ptr<uchar>(j);
            interpLinePtr = mod_diff.ptr<float>(j - 1);

            //first row
            resLinePtr[0] = imgLinePtr[0];

            for (int i = 1; i < w - 1; ++i) {
                //saturate cast is not used here, because it behaves differently from the previous implementation
                //most notable, saturate_cast rounds before truncating, here it's the opposite.
                float value = interpLinePtr[i - 1];
                if (value < 0.)
                    resLinePtr[i] = 0;
                else if (value > 255.0)
                    resLinePtr[i] = 255;
                else
                    resLinePtr[i] = static_cast<uchar>(value);
            }

            //last row
            resLinePtr[w - 1] = imgLinePtr[w - 1];
        }

        //last col
        resLinePtr = result.ptr<uchar>(h - 1);
        imgLinePtr = img.ptr<uchar>(h - 1);
        for (int i = 0; i < w; ++i)
            resLinePtr[i] = imgLinePtr[i];
    }

    void poissonSolver(const Mat &img, Mat &laplacianX, Mat &laplacianY, Mat &result) {
        const int w = img.cols;
        const int h = img.rows;

        Mat lap = Mat(img.size(), CV_32FC1);

        lap = laplacianX + laplacianY;

        Mat bound = img.clone();

        // Fills out everything but a 1-border on image with black.
        rectangle(bound, Point(1, 1), Point(img.cols - 2, img.rows - 2), Scalar::all(0), -1);
        Mat boundary_points;
        // Computes laplacian (center will be 0).
        Laplacian(bound, boundary_points, CV_32F);

        // somehow fixes up boundaries?
        boundary_points = lap - boundary_points;

        Mat mod_diff = boundary_points(Rect(1, 1, w - 2, h - 2));

        //mod_diff is the laplacian of the image with somewhat fixed boundaries.
        solve(img, mod_diff, result);
    }
}
#endif //PATCHMATCH_POISSON_H
