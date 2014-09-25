/*
 * cameraCalibration.h
 *
 *  Created on: Dec 12, 2013
 *      Author: ezwei
 */

#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifndef CAMERACALIBRATION_H_
#define CAMERACALIBRATION_H_

class cameraCalibration{
private:
	// private variables
	float alphaStart;
	float alphaRes;
	float alphaNum;
	bool selfTrainingMode;
	std::vector<cv::Point2f> imageCorners;
	std::vector<cv::Point2f> realCorners;
	std::vector<float> alphaList;
	// private functions
	void showCorners();
	void findEdges(int thrLow = 20, int thrHigh = 30);
public:
	// public variables
	cv::Mat H; // transformation Matrix
	cv::Mat frame;
	cv::Mat calibratedFrame;
	cv::Mat cannyOutput;
	bool calibrated;
	int w_Est;
	int i_Opt;
	// public functions
	double getLineCurvatureAngle();
	void getLateralOffset(int wmin, int wmax, int d, float alphaEst);
	void addCorner(cv::Point2f p);
	void calibrate();
	cameraCalibration(bool selfTraining, float Range, float Res);
	~cameraCalibration();
};

#endif /* CAMERACALIBRATION_H_ */
