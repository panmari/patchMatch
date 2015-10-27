#ifndef PATCHMATCH_UTIL_H
#define PATCHMATCH_UTIL_H

#include <boost/format.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

namespace pmutil {

    using boost::format;
    using cv::imwrite;
    using cv::Mat;
    using cv::Matx;
    using cv::Rect;
    using cv::Scalar;
    using cv::Size;
    using cv::String;
    using cv::Vec3f;

    /**
     * Computes the sum of squared differences of the two given matrices/images. Assumes that they have the same size
     * and type. 
     */
    static double ssd(const Mat &img, const Mat &img2) {
        Mat diff = img - img2;
        Mat squares = diff.mul(diff);
        Scalar ssd_channels = sum(squares);
        double ssd = ssd_channels[0] + ssd_channels[1] + ssd_channels[2];
        return ssd;
    }

    /**
	 * Same as ssd, but not safe and will access garbage if given the possibility.
	 * Only works for float matrices right now!
	 * TODO: extend for other types.
	 */
    static double ssd_unsafe(const Mat &img, const Mat &img2, double limit = INFINITY) {
		double ssd = 0.f;
		int nRows = img.rows;
		int nCols = img.cols * img.channels();
		// We can do even better if the values are in one huge block.
		if (img.isContinuous() && img2.isContinuous())
		{
			nCols *= nRows;
			nRows = 1;
		}
		for (int i = 0; i < nRows; i++) {
			const float *p1 = img.ptr<const float>(i);
			const float *p2 = img2.ptr<const float>(i);
			for (int j = 0; j < nCols; j++)
			{
				float diff = p1[j] - p2[j];
				ssd += diff * diff;
			}
            // If we're higher than previous limit, return prematurely (since we're only looking for minimum).
            if (ssd >= limit) {
                return ssd;
            }
		}
		return ssd;
	}

    /**
     * Convert images to lab retrieved from imread.
     * L*a*b has the following ranges for each channel:
     * L: [0, 100]
     * a*: [-170, 100]
     * b*: [-100, 150]
     */
    static void convert_for_computation(Mat &img, const float resize_factor) {
        if (resize_factor != 1.f) {
            resize(img, img, Size(), resize_factor, resize_factor);
        }
        img.convertTo(img, CV_32FC3, 1 / 255.f);
        cvtColor(img, img, CV_BGR2Lab);
    }

    /**
     * Writes the given img to a file with the given filename, converting it first from lab to bgr
     */
    static void imwrite_lab(const String filename, const Mat &img) {
        Mat bgr;
        cvtColor(img, bgr, CV_Lab2BGR);
        imwrite(filename, bgr);
    }

    /**
     * Dumps all nearest patches for visual inspection
     */
    static void dumpNearestPatches(Mat &offset_map, Mat &source, int patch_size, String filename_prefix) {
        for (int x = 0; x < offset_map.cols; x++) {
            for(int y = 0; y < offset_map.rows; y++) {
                Vec3f offset = offset_map.at<Vec3f>(y, x);
                Rect nearest_patch_rect(x + offset[0], y + offset[1], patch_size, patch_size);
                Mat nearest_patch = source(nearest_patch_rect);
                imwrite(str(format("patch_%s_x_%03d_y_%03d.exr") % filename_prefix % x % y), nearest_patch);
            }
        }
    }

    static void computeGradientX(const Mat &img, Mat &gx) {
        Mat kernel = Mat::zeros(1, 3, CV_8S);
        kernel.at<char>(0, 2) = 1;
        kernel.at<char>(0, 1) = -1;

        if (img.channels() == 3) {
            filter2D(img, gx, CV_32F, kernel);
        }
        else if (img.channels() == 1) {
            Mat tmp[3];
            for (int chan = 0; chan < 3; ++chan) {
                filter2D(img, tmp[chan], CV_32F, kernel);
            }
            merge(tmp, 3, gx);
        }
    }

    static void computeGradientY(const Mat &img, Mat &gy) {
        Mat kernel = Mat::zeros(3, 1, CV_8S);
        kernel.at<char>(2, 0) = 1;
        kernel.at<char>(1, 0) = -1;

        if (img.channels() == 3) {
            filter2D(img, gy, CV_32F, kernel);
        }
        else if (img.channels() == 1) {
            Mat tmp[3];
            for (int chan = 0; chan < 3; ++chan) {
                filter2D(img, tmp[chan], CV_32F, kernel);
            }
            merge(tmp, 3, gy);
        }
    }

    constexpr double MIN_SHIFT_DISTANCE = 0.01;
    constexpr double MIN_CLUSTER_DISTANCE = 0.1;
    constexpr double EPSILON = 1e-6;

    /**
     * Does mean shift on the given colors with a gaussian kernel (using the given sigma).
     * The found modes (cluster centers) are saved in the given modes array, the assignments in the other.
     *
     * @param mode_assignments returns a vector of the same size as 'colors',
     *   with values between 0 and size of 'modes' - 1.
     */
    static void naiveMeanShift(const std::vector<Vec3f> &colors, const double sigma,
                               std::vector<Vec3f> *modes, std::vector<int> *mode_assignments) {
        modes->resize(0);
        mode_assignments->resize(0);
        const double two_sigma_sqr = 2 * sigma * sigma;
        for (const Vec3f &color: colors) {
            Vec3f center(color);
            while (true) {
                Vec3f new_center = Vec3f(0, 0, 0);
                double total_weight = 0;
                for (const Vec3f &other_color: colors) {
                    auto dir = center - other_color;
                    double dist = cv::norm(dir);
                    double weight = exp(-dist / (two_sigma_sqr + EPSILON));
                    new_center += weight * other_color;
                    total_weight += weight;
                }
                new_center /= total_weight;
                if (norm(new_center, center) < MIN_SHIFT_DISTANCE)
                    break;
                center = new_center;
            }
            bool create_mode = true;
            for (int i = 0; i < modes->size(); i++) {
                const Vec3f &mode = modes->at(i);
                if (norm(center, mode) < MIN_CLUSTER_DISTANCE) {
                    mode_assignments->push_back(i);
                    create_mode = false;
                    break;
                }
            }
            if (create_mode) {
                mode_assignments->push_back(static_cast<int>(modes->size()));
                modes->push_back(center);
            }
        }
    }

}

#endif //PATCHMATCH_UTIL_H
