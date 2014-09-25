/*
 * main.cpp
 *
 *  Created on: Dec 8, 2013
 *      Author: ezwei
 */

#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cameraCalibration.h"

bool selfTraining = false;

cameraCalibration myCalib(selfTraining,CV_PI/3,5*CV_PI/180);

void onMouse(int event, int x, int y, int flags, void* param);

int main(){
	cv::VideoCapture capture("/home/ezwei/workspace/laneDepartureDetection/source/lane4.avi");
	std::vector<float> alphaList;
	std::vector<float> wEstList;
	std::vector<float> iOptList;
	cv::namedWindow("canny",1);
	cv::namedWindow("sourceFrame",1);

	if(!selfTraining)
		cv::setMouseCallback("sourceFrame", onMouse, 0);
	cv::Mat frame = cv::Mat::zeros(100,100,CV_8UC1);
	while(true){
		//capture>>myCalib.frame;
		capture>>frame;
		if(frame.empty()){
			std::cout<<"video bitti"<<std::endl;
			break;
		}
		cv::resize(frame,myCalib.frame,cv::Size(0,0),0.7,0.7,cv::INTER_NEAREST);
		cv::cvtColor(myCalib.frame,myCalib.frame,CV_BGR2GRAY); // Optional
		if(myCalib.calibrated){
			myCalib.calibrate();
			float alphaEst = myCalib.getLineCurvatureAngle();
			myCalib.getLateralOffset(20,200,100,alphaEst);
			alphaList.push_back(alphaEst);
			wEstList.push_back(myCalib.w_Est);
			iOptList.push_back(myCalib.i_Opt);
			cv::Mat screen;
			cvtColor(myCalib.calibratedFrame,screen,CV_GRAY2RGB);
			if(alphaList.size()>=10){
				alphaList.erase(alphaList.begin());
				wEstList.erase(wEstList.begin());
				iOptList.erase(iOptList.begin());
				float alphaAvr = cv::mean(cv::Mat(alphaList)).val[0];
				float wEstAvr = cv::mean(cv::Mat(wEstList)).val[0];
				float iOptAvr = cv::mean(cv::Mat(iOptList)).val[0];
				// center line
				//cv::line(screen, cv::Point(myCalib.frame.cols/2,myCalib.frame.rows),cv::Point(myCalib.frame.cols/2,0),cv::Scalar(255,0,0),1,4);
				// left lane line
				cv::line(screen, cv::Point(myCalib.frame.cols/2+iOptAvr-wEstAvr/2,myCalib.frame.rows),cv::Point(myCalib.frame.cols/2+iOptAvr-wEstAvr/2- myCalib.frame.rows/tan(CV_PI/2-alphaAvr),0),cv::Scalar(255,255,255),1,4);
				// right lane line
				cv::line(screen, cv::Point(myCalib.frame.cols/2+iOptAvr+wEstAvr/2,myCalib.frame.rows),cv::Point(myCalib.frame.cols/2+iOptAvr+wEstAvr/2- myCalib.frame.rows/tan(CV_PI/2-alphaAvr),0),cv::Scalar(255,255,255),1,4);
				// curvature line
				cv::line(screen, cv::Point(myCalib.frame.cols/2,myCalib.frame.rows),cv::Point(myCalib.frame.cols/2-myCalib.frame.rows/tan(CV_PI/2-alphaEst),0),cv::Scalar(0,0,255),1,4);
			}

			cv::imshow("canny", myCalib.cannyOutput);
			cv::imshow("Corrected", screen);
			//if(cv::waitKey(30) >= 0) break;
		}
		cv::imshow("sourceFrame",myCalib.frame);
		if(cv::waitKey(30) >= 0) break;
	}
	return 0;
}

void onMouse(int event, int x, int y, int flags, void* param){
    char text[100];
    cv::Mat img2;
    img2 = myCalib.frame.clone();
    if (event == CV_EVENT_LBUTTONDOWN && !myCalib.calibrated){
    	myCalib.addCorner(cv::Point2f(x,y));
    }
    else{
        sprintf(text, "x=%3d, y=%3d", x, y);
        cv::putText(img2, text, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.85, CV_RGB(255,255,255));
        cv::line(img2,cv::Point(x,0),cv::Point(x,480),CV_RGB(255,255,255),1,CV_AA);
        cv::line(img2,cv::Point(0,y),cv::Point(640,y),CV_RGB(255,255,255),1,CV_AA);
    }
    cv::imshow("sourceFrame",img2);
}
