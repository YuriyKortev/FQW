#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "cracks_detector.h"

#include <iostream>

MainWindow::MainWindow(QWidget* parent)
	: QMainWindow(parent)
	, ui(new Ui::MainWindow)
{
	ui->setupUi(this);
	Cracks_detector* model = new Cracks_detector("C:/Users/green/snv_reposes/converter/Sandbox/trunk/cracks_detector/frozen_graph.pb");
	/*
		cv::Mat frame = cv::imread("C:/test_images/13.png", cv::IMREAD_GRAYSCALE);

		cv::Mat processed_image = model->process_image(frame);
		cv::imshow("res", processed_image);
	**/
	 
	 // model->detect_dir_of_frames("C:/full_size_images/fouth/", "C:/full_size_images/res/");


	/*
	std::vector<cv::String> paths;
	cv::glob("C:/full_size_images/fouth/*.png", paths);

	std::vector<Crack_info> infos = model->get_cracks(paths, -217.0, 9108.0);

	for (auto crack_info : infos) {
		qDebug() << '{' << crack_info.top.x << ' ' << crack_info.top.y << '}' << ' ' << '{' << crack_info.bottom.x << ' ' << crack_info.bottom.y << '}' << ' ' << crack_info.width.max_width;
	}

	qDebug() << infos.size();
	*/

}

MainWindow::~MainWindow()
{
	delete ui;
}

