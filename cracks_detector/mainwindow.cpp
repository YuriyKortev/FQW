#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "cracks_detector.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    Cracks_detector* model = new Cracks_detector("C:/frozen_graph.pb");
/*
    cv::Mat frame = cv::imread("C:/Users/green/QtProjects/cracks_detector/test_images/13.png", cv::IMREAD_GRAYSCALE);

    cv::Mat processed_image = model->process_image(frame);
    cv::imshow("res", processed_image);
**/
    model->detect_dir_of_frames("C:/test_images/", "C:/test_res/");
}

MainWindow::~MainWindow()
{
    delete ui;
}

