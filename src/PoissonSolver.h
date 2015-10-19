// Taken from
// https://github.com/Itseez/opencv/blob/ddf82d0b154873510802ef75c53e628cd7b2cb13/modules/photo/src/seamless_cloning_impl.cpp
#ifndef PATCHMATCH_POISSON_H
#define PATCHMATCH_POISSON_H

#include "opencv2/imgproc/imgproc.hpp"

using cv::filter2D;
using cv::Mat;
using cv::Point;
using cv::Rect;
using cv::Scalar;

class PoissonSolver {
public:

    PoissonSolver(const Mat &img, const Mat &gradient_x, const Mat &gradient_y) : img(img) {
        const int w = img.cols;
        filter_X.resize(w - 2);
        for (int i = 0; i < w - 2; ++i)
            filter_X[i] = 2.0f * std::cos(static_cast<float>(CV_PI) * (i + 1) / (w - 1));

        const int h = img.rows;
        filter_Y.resize(h - 2);
        for (int j = 0; j < h - 2; ++j)
            filter_Y[j] = 2.0f * std::cos(static_cast<float>(CV_PI) * (j + 1) / (h - 1));

        computeLaplacian(gradient_x, gradient_y, lap);
    }

    void solve(Mat &result) {
        const int w = img.cols;
        const int h = img.rows;

        Mat bound = img.clone();

        // Fills out everything but a 1-border on image with black.
        rectangle(bound, Point(1, 1), Point(img.cols - 2, img.rows - 2), Scalar::all(0), -1);
        Mat boundary_points;
        // Computes laplacian (center will be 0).
        Laplacian(bound, boundary_points, CV_32F);

        // somehow fixes up boundaries?
        boundary_points = lap - boundary_points;

        //mod_diff is the laplacian of the image with somewhat fixed boundaries.
        Mat mod_diff = boundary_points(Rect(1, 1, w - 2, h - 2));

        // Every channel is solved for individually, so we split here into single parts and merge at the end.
        std::vector<Mat> img_chans;
        cv::split(img, img_chans);
        std::vector<Mat> mod_diff_chans;
        cv::split(mod_diff, mod_diff_chans);
        std::vector<Mat> result_chans;
        for(int c = 0; c < 3; c++) {
            Mat single_chan;
            single_chan.create(img.rows, img.cols, CV_32FC1);
            solve(img_chans[c], mod_diff_chans[c], single_chan);
            result_chans.push_back(single_chan);
        }
        cv::merge(result_chans, result);
    }

private:
    std::vector<float> filter_X, filter_Y;
    const cv::Mat img;
    cv::Mat lap;

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

    void solve(const Mat &img, Mat &mod_diff, Mat &result) {
        const int w = img.cols;
        const int h = img.rows;

        Mat res;
        dst(mod_diff, res);

        for (int j = 0; j < h - 2; j++) {
            float *resLinePtr = res.ptr<float>(j);
            for (int i = 0; i < w - 2; i++) {
                resLinePtr[i] /= (filter_X[i] + filter_Y[j] - 4);
            }
        }

        idst(res, mod_diff);

        float *resLinePtr = result.ptr<float>(0);
        const float *imgLinePtr = img.ptr<float>(0);
        const float *interpLinePtr = NULL;

        //first col
        for (int i = 0; i < w; ++i)
            result.ptr<float>(0)[i] = img.ptr<float>(0)[i];

        for (int j = 1; j < h - 1; ++j) {
            resLinePtr = result.ptr<float>(j);
            imgLinePtr = img.ptr<float>(j);
            interpLinePtr = mod_diff.ptr<float>(j - 1);

            //first row
            resLinePtr[0] = imgLinePtr[0];

            for (int i = 1; i < w - 1; ++i) {
                //saturate cast is not used here, because it behaves differently from the previous implementation
                //most notable, saturate_cast rounds before truncating, here it's the opposite.
                float value = interpLinePtr[i - 1];
                resLinePtr[i] = value;
            }

            //last row
            resLinePtr[w - 1] = imgLinePtr[w - 1];
        }

        //last col
        resLinePtr = result.ptr<float>(h - 1);
        imgLinePtr = img.ptr<float>(h - 1);
        for (int i = 0; i < w; ++i)
            resLinePtr[i] = imgLinePtr[i];
    }

    void computeLaplacianX(const Mat &img, Mat &laplacianX) {
        Mat kernel = Mat::zeros(1, 3, CV_8S);
        kernel.at<char>(0, 0) = -1;
        kernel.at<char>(0, 1) = 1;
        filter2D(img, laplacianX, CV_32F, kernel);
    }

    void computeLaplacianY(const Mat &img, Mat &laplacianY) {
        Mat kernel = Mat::zeros(3, 1, CV_8S);
        kernel.at<char>(0, 0) = -1;
        kernel.at<char>(1, 0) = 1;
        filter2D(img, laplacianY, CV_32F, kernel);
    }

    void computeLaplacian(const Mat &gradient_x, const Mat &gradient_y, Mat &laplacian) {
        computeLaplacianX(gradient_x, laplacian);
        Mat lap_y;
        computeLaplacianY(gradient_y, lap_y);
        laplacian += lap_y;
    }
};

#endif //PATCHMATCH_POISSON_H
