#include <opencv2\opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;

#ifndef __MEMORY__
#define __MEMORY__

class Memory{

public:
	int m_r;
	int m_c;
	int m_N;
	int m_nlev;
	
	vector<cv::Mat> m_images;
	vector<cv::Mat> m_imagesGray;
	vector<cv::Mat> m_channels;
	vector<cv::Mat> m_weight_expose;
	vector<cv::Mat> m_weight_saturat;
	vector<cv::Mat> m_DIYweightimages;

	vector<cv::Size> m_pyrSize;
	
	vector<cv::Mat> m_Pyr;
	vector<cv::Mat> m_pyrW;
	vector<cv::Mat> m_pyrI;
	vector<cv::Mat> m_pyrI_auxiliary;
	vector<cv::Mat> m_pyr4downsample_32FC1;
	vector<cv::Mat> m_pyr4downsample_32FC3;


	cv::Mat m_kernel, m_kernel_T;

	Memory(int r, int c, int N);
	void SetImages(vector<cv::Mat> &images);
	
	vector<cv::Mat> allocatePyramid(int type);

	void gaussian_pyramid(cv::Mat &I, vector<cv::Mat> &pyr);
	void downsample32FC1(cv::Mat &I, cv::Mat &R, int levelnumber);
	void downsample32FC3(cv::Mat &I, cv::Mat &R, int levelnumber);
	void upsample(cv::Mat &I, cv::Mat &R);
	void laplacian_pyramid(cv::Mat &I);
	cv::Mat repmatchannelx3(cv::Mat &single);
};

#endif