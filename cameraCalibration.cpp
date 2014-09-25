/*
 * cameraCalibration.cpp
 *
 *  Created on: Dec 12, 2013
 *      Author: ezwei
 */

#include "cameraCalibration.h"

// public functions
void cameraCalibration::getLateralOffset(int wmin,int wmax, int d, float alphaEst){
	findEdges();
	cv::Mat offsetMap = cannyOutput.clone();

	cv::Mat accu = cv::Mat::zeros(wmax-wmin,cannyOutput.cols-wmax,CV_64FC1);
	for(int x1 = wmax, sayac=0; x1<offsetMap.cols;x1++,sayac++){
		float y1 = cannyOutput.rows;
		float y0 = cannyOutput.rows-d;
		float x0 = x1+ (y0-y1)/tan(CV_PI/2-alphaEst);
		cv::Point p1 = cv::Point(x1,y1);
		cv::Point p0 = cv::Point(x0,y0);
		for(int j = wmin,col=0;j<wmax; j++,col++){
			cv::LineIterator it(offsetMap,p0,p1,4);
			cv::Point diff= cv::Point(j,0);
			double sums =0;
			for(int i = 0; i < it.count; i++, ++it){
				cv::Point p = it.pos() - diff;
				sums += ((double)**it * (double)offsetMap.at<uchar>(p));
				//std::cout<<"v1: "<<val1<<" v2: "<<val2<<" i: "<<i<<" j: "<<j<<" it.pos: "<<it.pos()<<" p: "<<p<<"p1:"<<p1<<"p0:"<<p0<<std::endl;
			}
			//std::cout<<sums<<std::endl;
			accu.at<double>(col,sayac) = sums;
		}
	}
	cv::Point maxLoc;
	double maxVal;
	cv::minMaxLoc(accu, NULL, &maxVal, NULL, &maxLoc,cv::Mat());
	w_Est = wmin+maxLoc.y;
	i_Opt = -(wmax+maxLoc.x)+offsetMap.cols/2;
	std::cout<<"frame.cols: "<<offsetMap.cols<<" "<<" i: "<<i_Opt<<" w: "<<w_Est<<" o: "<<i_Opt-w_Est/2<<std::endl;
}

double cameraCalibration::getLineCurvatureAngle(){
	std::vector<float> alphaList;
	cv::Mat accumulator = cv::Mat::zeros(alphaNum,frame.cols,CV_64FC1);
	cv::Mat norm = cv::Mat::zeros(alphaNum,1,CV_64FC1);

	for(float alpha=-alphaStart, j=0;alpha<=alphaStart;alpha+=alphaRes,j++){
		alphaList.push_back(alpha);
		//cv::Mat lines = cv::Mat::zeros(frame.rows,frame.cols,CV_8UC1);
		for(int x1=0;x1<=frame.cols;x1++){
			float x0 = x1-(frame.rows/tan(CV_PI/2-alpha));
			float y0 = 0;
			float y1 = frame.rows;
			cv::LineIterator it(calibratedFrame,cv::Point(x0,y0),cv::Point(x1,y1),4);
			double sum = 0;
			for(int i = 0; i < it.count; i++, ++it){
				sum += (double)**it;
				//lines.at<uchar>(it.pos()) = 255;
			}
			accumulator.at<double>((int)j,x1) = sum;
			//cv::imshow("lines",lines);
			//if(cv::waitKey(30) >= 0) break;
		}
	}
	cv::Mat kernel = (cv::Mat_<double>(1,3) << 0,-1,1);
	cv::Mat delta;

	cv::filter2D(accumulator, delta, -1, kernel);

	delta = cv::abs(delta);

	for(int satir=0;satir<accumulator.rows;satir++){
		cv::Scalar sums = cv::sum(delta.row(satir));
		norm.at<double>(satir,0) =sums.val[0];
	}

	cv::Point maxLoc;
	double maxVal;

	cv::minMaxLoc(norm, NULL, &maxVal, NULL, &maxLoc,cv::Mat());

	float alphaEst = alphaList[maxLoc.y];
	//std::cout<<"max: "<<maxVal<<" @ "<<maxLoc<<" alphaEst: "<<alphaEst<<std::endl;

	return alphaEst;
}

cameraCalibration::cameraCalibration(bool selfTraining, float Range, float Res){
	std::cout<<"Constructing cameraCalibration Class.."<<std::endl;
	selfTrainingMode = selfTraining;
	calibrated = false;
	alphaStart = Range;
	alphaRes = Res;
	alphaNum = 2*alphaStart/alphaRes + 1;
}

cameraCalibration::~cameraCalibration(){
	std::cout<<"Deconstructor of cameraCalibration Class.."<<std::endl;
}

void cameraCalibration::addCorner(cv::Point2f p){
	std::size_t size = imageCorners.size();
	if(size<4){
		imageCorners.push_back(p);
		size += 1;
		switch (size%2){
			case 1: realCorners.push_back(p);
		    break;
		    case 0: realCorners.push_back(cv::Point2f(realCorners.back().x,p.y));
		    break;
		}
		if(size==4){
			H = cv::getPerspectiveTransform(imageCorners,realCorners);
			calibrated = true;
			cv::namedWindow("Corrected",1);
		}
	}
	showCorners();
}

void cameraCalibration::calibrate(){
	cv::warpPerspective(frame, calibratedFrame, H, cv::Size(frame.cols,frame.rows));
}


// private functions
void cameraCalibration::showCorners(){
	std::cout<<"Image Corners: "<<cv::Mat(imageCorners)<<std::endl<<"Real  Corners: "<<cv::Mat(realCorners)<<std::endl;
}

void cameraCalibration::findEdges(int thrLow, int thrHigh){
	cv::blur(calibratedFrame, cannyOutput, cv::Size(5,5));
	cv::Canny(cannyOutput,cannyOutput,thrLow,thrHigh);
}
