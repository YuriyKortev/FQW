#ifndef CRACKS_DETECTOR_H
#define CRACKS_DETECTOR_H

#include <iomanip>
#include <sstream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <QDebug>
#include <QFile>
#include <QFileInfo>

struct Cracks_width
{
    float max_width;
    cv::Point2f max_width_point;
};

struct Crack_info
{
    cv::Point2f top;
    cv::Point2f bottom;
    Cracks_width width;
};

class Cracks_detector
{
public:
    Cracks_detector(std::string tf_graph);

    cv::Mat process_image(cv::Mat & input_image);
    void detect_single_frame(std::string frame_path, std::string dest_frame_path);
    void detect_dir_of_frames(std::string dir_path, std::string dest_dir_path);
    void detect_from_cam(int num_of_cam);

    std::vector<Crack_info> get_cracks(std::vector<std::string> paths, float left_azimuth, float top_high_mark);

private:
    cv::dnn::Net unet;

    cv::Mat blob;

    cv::Mat convert_scores(cv::Mat & scores);

    cv::Point2f convert_point(cv::Point2f pixel_coord, float x_coef, float y_coef, float top_high_mark, float left_azimuth);
    float get_coef(float left, float right, int num_of_pixels);

    void draw_objects(cv::Mat & image, cv::Mat & mask, cv::Scalar color, bool draw_width);
    Cracks_width get_cracks_width(cv::Mat& crack);

    void resize_contour(std::vector<cv::Point> & contour, float cx, float cy);

    float pxls2mm(int pxls);

    cv::Mat forward_frame(cv::Mat & input_image);

    void add_cracks(cv::Mat& frame, cv::Mat & cracks);

    std::vector<int64_t> m_perfValues;
};

cv::Point2f operator*(cv::Point2f& p1, cv::Point2f& p2);
cv::Point2f operator*(cv::Point& p1, cv::Point2f& p2);
cv::Point2f operator+(cv::Point2f& p1, cv::Point2f& p2);

#endif // CRACKS_DETECTOR_H
