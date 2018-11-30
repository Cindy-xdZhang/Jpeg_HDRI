#include "Memory.h"
#include <opencv2\opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2\imgproc\imgproc.hpp>

//#include "../Fusion/Pyramid.h"

using namespace cv;

Memory::Memory(int r,int c,int N){
	this->m_N = N;
	this->m_r = r;
	this->m_c = c;

	m_images.resize(N);
	m_imagesGray.resize(N);
	m_weight_expose.resize(N);
	m_weight_saturat.resize(N);
	for (int i = 0; i < N; i++){
		m_images[i].create(r, c, CV_32FC3);
		m_imagesGray[i].create(r, c, CV_32FC1);
		m_weight_expose[i].create(r, c, CV_32FC1);
		m_weight_saturat[i].create(r, c, CV_32FC1);
	}

	m_channels.push_back(cv::Mat::zeros(r, c, CV_32FC1));
	m_channels.push_back(cv::Mat::zeros(r, c, CV_32FC1));
	m_channels.push_back(cv::Mat::zeros(r, c, CV_32FC1));
	
	m_kernel = Mat::zeros(1, 5, CV_32FC1);
	m_kernel.at<float>(0, 0) = .0625;
	m_kernel.at<float>(0, 1) = .25;
	m_kernel.at<float>(0, 2) = .375;
	m_kernel.at<float>(0, 3) = .25;
	m_kernel.at<float>(0, 4) = .0625;

	m_kernel_T = m_kernel.t();

	//create empty pyramid
	m_Pyr = allocatePyramid(CV_32FC3);
	m_pyrI = allocatePyramid(CV_32FC3);
	m_pyrI_auxiliary = allocatePyramid(CV_32FC3);
	m_pyrW = allocatePyramid(CV_32FC1);

	m_pyr4downsample_32FC1 = allocatePyramid(CV_32FC1);
	m_pyr4downsample_32FC3 = allocatePyramid(CV_32FC3);
	
	m_nlev = m_Pyr.size();
}

void Memory::SetImages(vector<cv::Mat> &images){
	for (int i = 0; i < m_N; i++){
		images[i].convertTo(m_images[i], CV_32FC3);
		m_images[i] /= 255;
		cv::cvtColor(images[i], m_imagesGray[i], CV_BGR2GRAY);
	}
}

void Memory::gaussian_pyramid(cv::Mat &I, vector<cv::Mat> &pyra){
	I.copyTo(pyra[0]);
	for (int i = 1; i < pyra.size(); i++)
		downsample32FC1(pyra[i - 1], pyra[i],i);
}

void Memory::downsample32FC1(cv::Mat &I, cv::Mat &R, int levelnumber){
	cv::GaussianBlur(I, m_pyr4downsample_32FC1[levelnumber], cv::Size(5, 5), 0, 0);
	cv::resize(m_pyr4downsample_32FC1[levelnumber], R, R.size(), 0, 0, INTER_NEAREST);
	
}

void Memory::downsample32FC3(cv::Mat &I, cv::Mat &R, int levelnumber){
	cv::GaussianBlur(I, m_pyr4downsample_32FC3[levelnumber], cv::Size(5, 5), 0, 0);
	cv::resize(m_pyr4downsample_32FC3[levelnumber], R, R.size(), 0, 0, INTER_NEAREST);
}

void Memory::upsample(cv::Mat &I, cv::Mat &R){
	cv::resize(I, R, R.size(), 0, 0, INTER_NEAREST);
	cv::GaussianBlur(R, R, cv::Size(5, 5), 0, 0);
}

void Memory::laplacian_pyramid(cv::Mat &I){
	cv::Mat J = I;
	for (int i = 0; i < m_nlev - 1; i++){
		downsample32FC3(J, m_pyrI_auxiliary[i + 1],i);
		upsample(m_pyrI_auxiliary[i + 1], m_pyrI[i]);
		m_pyrI[i] = J - m_pyrI[i];
		J = m_pyrI_auxiliary[i + 1];
	}
	J.copyTo(m_pyrI[m_nlev - 1]);
}

cv::Mat Memory::repmatchannelx3(cv::Mat &single){
	cv::Mat result;

	vector<cv::Mat> channels;
	channels.push_back(single);
	channels.push_back(single);
	channels.push_back(single);
	merge(channels, result);

	return result;
}

vector<cv::Mat> Memory::allocatePyramid(int type){
	vector<cv::Mat> pyr;
	
	int m = 0;
	m_c < m_r ? m = m_c : m = m_r;
	int nlev = floor(log(m) / log(2));
	int c = m_c;
	int r = m_r;

	pyr.push_back(cv::Mat::zeros(r,c,type));
	for (int i = 0; i < nlev - 1; i++){
		double rr = 1.0*r;
		double cc = 1.0*c;
		rr /= 2;
		cc /= 2;
		r = int(rr + .5);
		c = int(cc + .5);
		pyr.push_back(cv::Mat::zeros(r, c, type));
	}

	return pyr;
}