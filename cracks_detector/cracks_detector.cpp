#include "cracks_detector.h"

#include <chrono>

Cracks_detector::Cracks_detector(std::string tf_graph)
{
    this->unet = cv::dnn::readNet(tf_graph);
    unet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    unet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

cv::Mat Cracks_detector::process_image(cv::Mat & input_image)
{
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat mask = forward_frame(input_image);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    m_perfValues.push_back(elapsed.count());

    //создание копии исходного кадра в форма RGB для сохранения контуров дефектов
    cv::Mat processed_image = input_image.clone();
    cv::cvtColor(processed_image, processed_image, cv::COLOR_GRAY2RGB);

    //создание маски с трещинами
    cv::Mat defects = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
    defects.setTo(255, mask==1);

    //отрисовка контуров и ширины трещин
    draw_objects(processed_image, defects, cv::Scalar(0, 0, 255), true);

    //создание маски зазоров
    defects = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
    defects.setTo(255, mask==2);

    //отрисовка контуров зазоров
    draw_objects(processed_image, defects, cv::Scalar(0, 255, 0), false);

    return processed_image;
}

cv::Mat Cracks_detector::convert_scores(cv::Mat &scores)
{
    const int rows = scores.size[2];
    const int cols = scores.size[3];
    const int chnls = scores.size[1];

    cv::Mat maxVal(rows, cols, CV_32FC1, scores.data);
    cv::Mat mask = cv::Mat::zeros(rows, cols, CV_8UC1);

    for(int ch=0; ch < chnls; ch++){
        for(int row=0; row < rows; row++){
            const float* rowScore = scores.ptr<float>(0, ch, row);
            float* rowMax = maxVal.ptr<float>(row);
            uint8_t* rowMask = mask.ptr<uint8_t>(row);

            for(int col=0; col < cols; col++){
                //номер канала с максимальным скором - итоговый класс пикселя
                if(rowScore[col] > rowMax[col]){
                    rowMax[col] = rowScore[col];
                    rowMask[col] = (uchar)ch;
                }
            }
        }
    }

    return mask;
}

cv::Point2f Cracks_detector::convert_point(cv::Point2f pixel_coord, float x_coef, float y_coef, float top_high_mark, float left_azimuth)
{
    return cv::Point2f(pixel_coord.x * x_coef + left_azimuth, top_high_mark - pixel_coord.y * y_coef);
}

float Cracks_detector::get_coef(float left, float right, int num_of_pixels)
{
    return (right - left) / (float)num_of_pixels;
}

void Cracks_detector::draw_objects(cv::Mat & image, cv::Mat & mask, cv::Scalar color, bool draw_width)
{
    //коэффициенты масштабирования для контуров
    float cx = (float)image.cols / mask.cols;
    float cy = (float)image.rows / mask.rows;

    //получение контуров каждого объекта
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    for(size_t i=0; i < contours.size(); i++){
        resize_contour(contours[i], cx, cy);

        if(cv::contourArea(contours[i]) < 1500) continue;

        cv::drawContours(image, contours, (int)i, color, 2);

        if(!draw_width) continue;

        cv::Mat crack = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
        cv::drawContours(crack, contours, (int)i, cv::Scalar(255), -1);

        Cracks_width w = get_cracks_width(crack);
        uint max_width = w.max_width;
        cv::Point max_Point = w.max_width_point;

        cv::line(image, max_Point, cv::Point(max_Point.x + max_width, max_Point.y), cv::Scalar(0, 239, 250), 3);

        std::stringstream ss;
        ss << "Cracks width: " << std::setprecision(2) << pxls2mm(max_width) << " mm";

        std::string text_width = ss.str();

        int baseline;
        cv::Size text_size = cv::getTextSize(text_width, cv::FONT_HERSHEY_COMPLEX, 1, 1, &baseline);

        cv::Point text_point = cv::Point(max_Point.x + int(max_width / 2) - int(text_size.width / 2), max_Point.y - 15);

        if(text_point.y <= text_size.height){
            text_point.y = text_point.y + text_size.height + 70;
        }
        if(text_point.x < 0){
            text_point.x = 0;
        }else if(text_point.x + text_size.width > image.cols){
            text_point.x = image.cols - text_size.width;
        }

        cv::putText(image, text_width, text_point, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 239, 250), 1);

    }
}

Cracks_width Cracks_detector::get_cracks_width(cv::Mat& crack)
{
    cv::Point max_Point(0, 0);
    size_t max_width = 0;

    //нахождение максимальной ширины
    for (int row = 0; row < crack.rows; row++) {
        uint8_t* rowCrack = crack.ptr<uint8_t>(row);

        std::vector<cv::Point> locations;
        cv::findNonZero(cv::Mat(1, crack.cols, CV_8UC1, rowCrack), locations);

        if (locations.size() > max_width) {
            max_width = locations.size();
            max_Point.x = locations.front().x;
            max_Point.y = row;
        }
    }

    return { (float)max_width, max_Point };
}

void Cracks_detector::resize_contour(std::vector<cv::Point> & contour, float cx, float cy)
{
    for(cv::Point & p : contour){
        p.x = p.x * cx;
        p.y = p.y * cy;
    }
}

float Cracks_detector::pxls2mm(int pxls)
{
    return pxls * 0.0242;
}

cv::Mat Cracks_detector::forward_frame(cv::Mat& input_image)
{
    //обработка изображения: масштабирование к форме входного слоя, нормализация, BGR к RGB
    cv::dnn::blobFromImage(input_image, this->blob, 1 / 255., cv::Size(1984, 544), cv::Scalar(0, 0, 0), true, false);

    //закрашивание черных областей на изображении
    float mean = cv::mean(blob).val[0];
    cv::Mat black_areas = blob < 0.001;
    blob.setTo(mean, black_areas);

    //получение результата модели
    unet.setInput(this->blob);
    cv::Mat scores = unet.forward();

    //получение масок для каждого класса из выхода модели
    cv::Mat mask = convert_scores(scores);

	//присваивание черных областей фону
    black_areas = black_areas.reshape(mask.channels(), mask.dims, mask.size.p);
    mask.setTo(0, black_areas);

    return mask;
}

void Cracks_detector::add_cracks(cv::Mat& frame, cv::Mat& cracks)
{
    cv::Mat mask = forward_frame(frame);

    cv::Mat cracks_on_frame = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
    cracks_on_frame.setTo(255, mask == 1);

	//склеивание масок с трещинами
    cracks_on_frame.push_back(cracks);
    cracks = cracks_on_frame;
}

void Cracks_detector::detect_from_cam(int num_of_cam)
{
    //detection from camera, num_of_cam - number of device, esc - exit loop

    cv::VideoCapture cap;
    if(!cap.open(num_of_cam)){
        qDebug()<<"Can't find camera";
    }
    while(!(cv::waitKey(10) == 27)){
        cv::Mat frame;
        cap >> frame;

        cv::Mat output_frame = this->process_image(frame);

        cv::imshow("Camera", output_frame);
    }
}

std::vector<Crack_info> Cracks_detector::get_cracks(std::vector<std::string> paths, float left_azimuth, float top_high_mark)
{
    cv::Mat cracks;
    uint rows = 0;
    uint cols = 0;

    cv::Mat input_frame = cv::imread(paths.front(), cv::IMREAD_GRAYSCALE);
    rows += input_frame.rows;
    cols = input_frame.cols;

    add_cracks(input_frame, cracks);

    for (auto cur = std::next(paths.begin()); cur != paths.end(); ++cur) {
        input_frame = cv::imread(*cur, cv::IMREAD_GRAYSCALE);

        rows += input_frame.rows;

        add_cracks(input_frame, cracks);

        qDebug() << cracks.rows << (*cur).data();
    }

    std::vector<std::vector<cv::Point>> cracks_contours;
    cv::findContours(cracks, cracks_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    auto point_cmp = [](const cv::Point& lhs, const cv::Point& rhs) { return lhs.y < rhs.y; };

    std::vector<Crack_info> cracks_infos;

	//коэффициенты масштабирования контуров
    cv::Point2f coefs((float)cols / cracks.cols, (float)rows / cracks.rows);

    float coef_mm = get_coef(0, (float)paths.size() * 100.0, rows);
    float coef_degree = get_coef(left_azimuth, -left_azimuth, cols);

	//коэффициенты перевода в мм и углы
    cv::Point2f degree_mm_coefs = cv::Point2f(coef_degree, -coef_mm);
	//начальные значения мм и углов
    cv::Point2f tr_coefs = cv::Point2f(left_azimuth, top_high_mark);
    
    for (size_t i = 0; i < cracks_contours.size(); i++) {
        if (cv::contourArea(cracks_contours[i]) < 4000) continue;

        cracks.setTo(0);
        cv::drawContours(cracks, cracks_contours, (int)i, cv::Scalar(255), -1);

        Cracks_width w = get_cracks_width(cracks);
        w.max_width *= coefs.x * coef_mm;
        w.max_width_point = w.max_width_point * coefs * degree_mm_coefs + tr_coefs;

        auto topbottom_point = std::minmax_element(cracks_contours[i].begin(), cracks_contours[i].end(), point_cmp);
        cv::Point2f top = *topbottom_point.first * coefs * degree_mm_coefs + tr_coefs;
        cv::Point2f bottom = *topbottom_point.second * coefs * degree_mm_coefs + tr_coefs;

        Crack_info info = { top, bottom, w };
        cracks_infos.push_back(info);
    }

    return cracks_infos;
}

void Cracks_detector::detect_dir_of_frames(std::string dir_path, std::string dest_dir_path)
{
    //detect_single_frame for each frame in dir_path, store results in dest_dir_path

    std::vector<cv::String> paths;
    cv::glob(dir_path + "/*.png", paths);

    m_perfValues.clear();

    for(std::string path : paths){
        QFile f(QString::fromUtf8(path.c_str()));
        QFileInfo fileInfo(f.fileName());
        QString filename(fileInfo.fileName());

        std::string dest_path = dest_dir_path + filename.toLocal8Bit().constData();
        this->detect_single_frame(path, dest_path);
    }

    qDebug() << "full time = " << std::accumulate(std::cbegin(m_perfValues), std::cend(m_perfValues), 0ull) << ' ' << m_perfValues.size();
}

void Cracks_detector::detect_single_frame(std::string frame_path, std::string dest_frame_path)
{
    //read frame from frame_path, process and store result to dest_frame_path


    cv::Mat image = cv::imread(frame_path, cv::IMREAD_GRAYSCALE);
    
    cv::Mat detectedFrame = this->process_image(image);

    cv::imwrite(dest_frame_path, detectedFrame);

    qDebug() << frame_path.data() << ' '<< "detected";
}

cv::Point2f operator*(cv::Point2f& p1, cv::Point2f& p2)
{
    return cv::Point2f(p1.x * p2.x, p1.y * p2.y);
}

cv::Point2f operator*(cv::Point& p1, cv::Point2f& p2)
{
    cv::Point2f temp(p1);
    return temp * p2;
}

cv::Point2f operator+(cv::Point2f& p1, cv::Point2f& p2)
{
    return cv::Point2f(p1.x + p2.x, p1.y + p2.y);
}
