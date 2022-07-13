#include "remap.h"
#include <chrono>

double deg2rad(double deg)
{
    return deg * M_PI / 180.0;
}

void rotateImage(cv::Mat &src, cv::Mat &dst, double angle, cv::Point2d center)
{

    // get rotation matrix for rotating the image around its center in pixel coordinates
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle, center not relevant
    // cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
    // adjust transformation matrix
    // rot.at<double>(0, 2) += bbox.width / 2.0 - src.cols / 2.0;
    // rot.at<double>(1, 2) += bbox.height / 2.0 - src.rows / 2.0;

    cv::warpAffine(src, dst, rot, src.size(), cv::INTER_CUBIC);
}

cv::Mat preprocessImg(cv::Mat srcImg)
{
    cv::Mat croppedImg = srcImg(cv::Range(80, srcImg.rows), cv::Range(310, 1450));
    return croppedImg;
}

ScaraOcamModel::ScaraOcamModel(/* args */)
{
}

ScaraOcamModel::~ScaraOcamModel()
{
}

cv::Point2d ScaraOcamModel::world2cam(cv::Point3d &point3D)
{
    /*
    Projects 3D point into the image pixel

    Input:
        point3D (list) : [x, y, z] coordinate of 3d points
    Output:
        u (float) : row pixel coordinate
        v (float) : column pixel coordinate
    */

    int length_invpol = sizeof(invpol) / sizeof(invpol[0]);
    // std::cout << "length_invpol: " << length_invpol << std::endl;

    double norm = sqrt(point3D.x * point3D.x + point3D.y * point3D.y);
    double theta = atan(point3D.z / norm);

    cv::Point2d point2D;

    if (norm != 0)
    {
        double invnorm = 1 / norm;
        double t = theta;
        double rho = invpol[0];
        double t_i = 1;

        for (int i = 1; i < length_invpol; i++)
        {
            t_i *= t;
            rho += t_i * invpol[i];
        }

        double x = point3D.x * invnorm * rho;
        double y = point3D.y * invnorm * rho;

        point2D.y = c * x + d * y + yc;
        point2D.x = e * x + y + xc;
    }
    else
    {
        point2D.y = yc;
        point2D.x = xc;
    }
    return point2D;
}

void ScaraOcamModel::create_LUT_90(double R, double H, cv::Mat &mapx, cv::Mat &mapy)
{
    /*
    Create Look Up Table (LUT) for remapping first quadrant omni image into rectanble image
        The LUT only refer to 90 deg FOV of the top-right part (first quadrant) according to the center xc, yc

        (0, 0)
         _________________
        |        | / / / |
        |        | / / / |
        | (xc,yc)| / / / |
        ---------+--------
        |        |       |
        |        |       |
        |        |       |
        ------------------

        Input:
            R (float/int) : The radius of projection cylinder.
            H (float/int) : The height of output image.
        Output:
            mapx (numpy.ndarray) : mapx to be used in np.remap
            mapy (numpy.ndarray) : mapy to be used in np.remap
    */

    double sqrt2 = sqrt(2);

    for (int i = 0; i < int(H); i++)
    {
        for (int j = 0; j < int(2 * R); j++)
        {
            // #Transform the dst image coordinate to XYZ coordinate
            cv::Point3d point3d;

            point3d.y = sqrt2 / 2 * j;
            point3d.x = -(sqrt2 * R - sqrt2 / 2 * j);
            point3d.z = H / 2 - i;

            // #Reproject onto image pixel coordinate
            cv::Point2d point2d = world2cam(point3d);
            mapy.at<float>(i, j) = (float)point2d.y;
            mapx.at<float>(i, j) = (float)(point2d.x - xc); // Top-right, so translate x by xc
        }
    }
}

void ScaraOcamModel::create_panoramic_undistortion_LUT(const double Rmax, const double Rmin, cv::Mat &mapx, cv::Mat &mapy)
{
    int width = mapx.cols, height = mapx.rows;
    std::cout << "Height: " << height << " Width: " << width << std::endl;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            double theta = -((double)j) / (double)width * 2 * M_PI; // Note, if you would like to flip the image, just inverte the sign of theta
            double rho = Rmax - (Rmax - Rmin) / (double)height * (double)i;
            mapx.at<float>(i, j) = (float)(xc + rho * sin(theta));
            mapy.at<float>(i, j) = (float)(yc + rho * cos(theta));
        }
    }
}

cv::Mat ScaraOcamModel::panoramic_rectify(cv::Mat src, int Rmax, int Rmin, cv::Size newImgSize)
{
    if (LUT_pan.size() < 2)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        cv::Mat mapx(newImgSize, CV_32FC1);
        cv::Mat mapy(newImgSize, CV_32FC1);

        create_panoramic_undistortion_LUT(Rmax, Rmin, mapx, mapy);
        LUT_pan.push_back(mapx);
        LUT_pan.push_back(mapy);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> time_ms = t2 - t1;
        std::cout << "LUT pan created!"
                  << "Used " << time_ms.count() << " ms." << std::endl;
    }

    cv::Mat mapx = LUT_pan[0];
    cv::Mat mapy = LUT_pan[1];
    cv::Mat dst(newImgSize, src.type());
    cv::Mat rectified(newImgSize, src.type());

    // std::cout << mapx.at<float>(20, 50) << std::endl;
    cv::remap(src, dst, mapx, mapy, cv::INTER_CUBIC);

    // cv::rotate(dst, rectified, cv::ROTATE_180);

    return dst;
}

cv::Mat ScaraOcamModel::cuboid_rectify(cv::Mat src, std::vector<cv::Mat> &imgs)
{
    /*
    Rectify omnidirectional image into panoramic image

    Return the perspective images of front, right, back, left and concatenation of four images

    Input:
        src (numpy.ndarray) : Input omnidirectional image
    Output:
        imgs (list[numpy.ndarray]) : Perspective images
        all_image (numpy.ndarray) : Concatenated image
    */
    imgs.clear();

    cv::Mat front;
    // # Rotate to align front to the middle
    cv::Mat rotated;
    rotateImage(src, rotated, 225, center);
    // cv::imshow("rotated", rotated);

    if (LUT_90.size() < 2)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        // #The radius of projection cylinder
        double R = std::min(xc, yc);
        // #The height of projection cylinder
        // #assuming 30 degrees fov above and below horizon of lens O
        double H = R * tan(deg2rad(30)) * 2;
        cv::Size2d newImgSize(2 * R, H);
        cv::Mat mapx(newImgSize, CV_32FC1);
        cv::Mat mapy(newImgSize, CV_32FC1);

        create_LUT_90(R, H, mapx, mapy);
        LUT_90.push_back(mapx);
        LUT_90.push_back(mapy);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> time_ms = t2 - t1;
        std::cout << "LUT 90 created!"
                  << "Used " << time_ms.count() << " ms." << std::endl;
    }

    cv::Mat mapx = LUT_90[0];
    cv::Mat mapy = LUT_90[1];

    cv::Mat all_img;

    // #For each iteration, project the top - right part(first quadrant) according to the center xc, yc
    // #Then rotate the image by 90 deg to get the next perspective
    for (int i = 0; i < 4; i++)
    {
        front = rotated(cv::Range(0, 489), cv::Range(608, rotated.cols));
        // cv::imshow("front", front);
        // cv::waitKey();

        // cv::Mat front = src;
        cv::Mat res_90;
        cv::remap(front, res_90, mapx, mapy, cv::INTER_CUBIC);
        imgs.push_back(res_90);
        rotateImage(rotated, rotated, 90, center);
    }
    cv::hconcat(imgs[0], imgs[1], all_img);
    cv::hconcat(all_img, imgs[2], all_img);
    cv::hconcat(all_img, imgs[3], all_img);
    // all_img = np.concatenate(imgs, axis = 1)
    return all_img;
}

MeiOcamModel::MeiOcamModel(/* args */)
{
}

MeiOcamModel::~MeiOcamModel()
{
}