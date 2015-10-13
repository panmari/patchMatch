#include "opencv2/ts.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

TEST(performance_test, performance_test_division_of_3d_by_1d) {
    vector<Size> sizes{Size(2, 2), Size(10, 10), Size(100, 100), Size(1000, 1000), Size(2000, 2000)};

    cout << "Size \t\tMethod 1 \tMethod 2 \tMethod 3" << "\tMethod 4" << endl;

    for (int is = 0; is < sizes.size(); ++is) {

        Size sz = sizes[is];
        Mat weighted_sum(sz, CV_32FC3);
        randu(weighted_sum, 0.0, 200.0);

        Mat weights(sz, CV_32FC1);
        randu(weights, 1.0, 10.0);

        Mat ws1 = weighted_sum.clone();
        Mat ws2 = weighted_sum.clone();
        Mat ws3 = weighted_sum.clone();
        Mat ws4 = weighted_sum.clone();

        // Method 1 @panmari
        double tic1 = double(getTickCount());
        Mat rec1;
        vector<Mat> channels(3);
        split(ws1, channels);
        for (Mat chan : channels) {
            divide(chan, weights, chan);
        }
        merge(channels, rec1);

        double toc1 = (double(getTickCount() - tic1)) * 1000. / getTickFrequency();

        // Method 2 @Miki
        double tic2 = double(getTickCount());
        Mat rec2 = ws2.reshape(3, 1);
        Mat ww = weights.reshape(1, 1);
        for (int i = 0; i < rec2.cols; ++i) {
            float w = ww.at<float>(0, i);
            Vec3f *v = rec2.ptr<Vec3f>(0, i);
            v->val[0] /= w;
            v->val[1] /= w;
            v->val[2] /= w;
        }
        rec2 = rec2.reshape(3, ws2.rows);

        double toc2 = (double(getTickCount() - tic2)) * 1000. / getTickFrequency();

        // Method 3 @Miki (+ @Micka)
        double tic3 = double(getTickCount());
        Mat3f rec3 = ws3.reshape(3, 1);
        //Mat3f rec3 = ws3.reshape(3, 1).clone(); // To not override original image
        Mat1f ww3 = weights.reshape(1, 1);

        Vec3f* prec3 = rec3.ptr<Vec3f>(0);
        float* pww = ww3.ptr<float>(0);

        for (int i = 0; i < rec3.cols; ++i)
        {
            float scale = 1. / (*pww);
            (*prec3)[0] *= scale;
            (*prec3)[1] *= scale;
            (*prec3)[2] *= scale;

            ++prec3; ++pww;
        }
        rec3 = rec3.reshape(3, ws3.rows);

        double toc3 = (double(getTickCount() - tic3)) * 1000. / getTickFrequency();

        // Method 4 @Micka
        double tic4 = double(getTickCount());
        Mat3f rec4;
        Mat3f w3ch;
        cvtColor(weights, w3ch, COLOR_GRAY2BGR);
        divide(ws4, w3ch, rec4);

        double toc4 = (double(getTickCount() - tic4)) * 1000. / getTickFrequency();

        cout << sz << " \t" << toc1 << " \t" << toc2 << " \t" << toc3 << " \t" << toc4 << endl;

        // Check for equality of methods.
        Mat diff;
        absdiff(rec1, rec2, diff);
        EXPECT_EQ(0, countNonZero(diff.reshape(1)));

        absdiff(rec1, rec3, diff);
        threshold(diff, diff, 1e-4, 1, THRESH_BINARY);
        EXPECT_EQ(0, countNonZero(diff.reshape(1)));

        absdiff(rec1, rec4, diff);
        EXPECT_EQ(0, countNonZero(diff.reshape(1)));
    }
}
