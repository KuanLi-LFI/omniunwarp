#include "remap.h"
#include <glob.h>   // glob(), globfree()
#include <string.h> // memset()
#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>
#include <chrono>

std::vector<std::string> glob(const std::string &pattern)
{
    using namespace std;

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if (return_value != 0)
    {
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    vector<string> filenames;
    for (size_t i = 0; i < glob_result.gl_pathc; ++i)
    {
        filenames.push_back(string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: omniUnwrap <image_path>" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(argv[1]);
    cv::Mat cropped = preprocessImg(img);
    std::vector<cv::Mat> imgs;

    std::vector<std::string> fnames = glob("./test/*.jpg");
    for (auto s : fnames)
    {
        // std::cout << s << std::endl;
        img = cv::imread(s);
        imgs.push_back(preprocessImg(img));
    }

    ScaraOcamModel scara;
    cv::Size newImgSize(1800, 400);
    cv::Mat panImg;
    cv::Mat cubImg;
    std::vector<cv::Mat> perImgs;
    panImg = scara.panoramic_rectify(cropped, 540, 200, newImgSize);
    // cv::imshow("panoramic", panImg);

    // cubImg = scara.cuboid_rectify(cropped, perImgs);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (auto img : imgs)
    {
        // cubImg = scara.cuboid_rectify(img, perImgs);
        panImg = scara.panoramic_rectify(img, 540, 200, newImgSize);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_ms = t2 - t1;
    std::cout << "average time per image in ms: " << time_ms.count() / 100 << std::endl;
    // cv::imshow("cuboid", cubImg);

    // cv::waitKey();

    return 0;
}