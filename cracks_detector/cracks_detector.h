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

class Cracks_detector
{
public:
    Cracks_detector(std::string tf_graph);

    cv::Mat process_image(cv::Mat & input_image);
    void detect_single_frame(std::string frame_path, std::string dest_frame_path);
    void detect_dir_of_frames(std::string dir_path, std::string dest_dir_path);
    void detect_from_cam(size_t num_of_cam);

private:
    cv::dnn::Net unet;

    cv::Mat blob;
    cv::Mat scores;

    cv::Mat convert_scores(cv::Mat & scores);
    void draw_objects(cv::Mat & image, cv::Mat & mask, cv::Scalar color, bool draw_width);

    void resize_contour(std::vector<cv::Point> & contour, float cx, float cy);

    float pxls2mm(int pxls);
};

#endif // CRACKS_DETECTOR_H
