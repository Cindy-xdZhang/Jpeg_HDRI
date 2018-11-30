#include "Memory.h"

#ifndef __MULTIFUSION__
#define __MULTIFUSION__

class MultiFusion{
private:
	Memory* m_memory;
	cv::Mat m_result;
private:
	void Contrast();
	void Saturation();
	void WellExposedness();
	void MakeWeights();
	void MakeWeights(vector<cv::Mat> &masks);
	void reconstruct_laplacian_pyramid();
public:
	MultiFusion(vector<cv::Mat> &rgb);
	~MultiFusion();
	void Execute(vector<cv::Mat> &masks);
	cv::Mat GetResult(){ return m_result; }
};
#endif