#ifndef REMAP_H
#define REMAP_H

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ccalib/omnidir.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#define CMV_MAX_BUF 1024
#define MAX_POL_LENGTH 64

void rotateImage(cv::Mat &src, cv::Mat &dst, double angle, cv::Point2d center);

class ScaraOcamModel
{
private:
    /* data */
    double yc = 489.624522, xc = 608.819613; // coordinate of the center, row and column
    cv::Point2d center = cv::Point2d(608.819613, 489.624522);
    double c = 0.998454, d = -0.008966, e = -0.008322; // affine parameter
    // the coefficients of the inverse polynomial
    double invpol[17] = {323.712933, 300.441941, 91.808710, -51.905017, -80.453176, 63.731282, 130.714839, -23.557147, -133.764001, -14.408450, 91.162738, 31.999104, -32.157217, -18.885584, 3.285893, 4.131682, 0.746373};
    // the polynomial coefficients: pol[0] + x"pol[1] + x^2*pol[2] + ... + x^(N-1)*pol[N-1]
    double ss[5] = {-1.864565e+02, 0.000000e+00, 2.919291e-03, -6.330598e-06, 8.678134e-09};

    // LUT cuboid
    std::vector<cv::Mat> LUT_90;
    // LUT panoramic
    std::vector<cv::Mat> LUT_pan;

public:
    ScaraOcamModel(/* args */);
    ~ScaraOcamModel();

    void get_ocam_model(std::string filename);
    void create_panoramic_undistortion_LUT(const double Rmin, const double Rmax, cv::Mat &mapx, cv::Mat &mapy);
    void create_LUT_90(double R, double H, cv::Mat &mapx, cv::Mat &mapy);
    cv::Point2d world2cam(cv::Point3d &point3D);
    cv::Mat panoramic_rectify(cv::Mat src, int Rmax, int Rmin, cv::Size newImgSize);
    cv::Mat cuboid_rectify(cv::Mat src, std::vector<cv::Mat> &imgs);
};

class MeiOcamModel
{
private:
    /* data */
    cv::Mat K;
    cv::Mat D;
    cv::Mat Xi;

public:
    MeiOcamModel(/* args */);
    ~MeiOcamModel();

    void get_ocam_model(std::string filename);
    cv::Mat panoramic_rectify(cv::Mat src, cv::Size new_img_size);
};

cv::Mat preprocessImg(cv::Mat srcImg);

#endif