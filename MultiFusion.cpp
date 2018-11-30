#include <opencv2\opencv.hpp>
#include "MultiFusion.h"

using namespace cv;

MultiFusion::MultiFusion(vector<cv::Mat> &rgb){

	int r = rgb[0].rows;
	int c = rgb[0].cols;
	int N = rgb.size();

	m_memory = new Memory(r, c, N);
	m_memory->SetImages(rgb);
}

void MultiFusion::Contrast(){
	
	cv::Mat kernel = cv::Mat::zeros(3, 3, CV_32FC1);
	kernel.at<float>(0, 1) = 1;
	kernel.at<float>(1, 0) = 1;
	kernel.at<float>(1, 1) = -4;
	kernel.at<float>(1, 2) = 1;
	kernel.at<float>(2, 1) = 1;

	for (int i = 0; i < m_memory->m_N; i++){
		filter2D(m_memory->m_imagesGray[i], m_memory->m_imagesGray[i], CV_32FC1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
		m_memory->m_imagesGray[i] = cv::abs(m_memory->m_imagesGray[i]);
	}
	
	double avg = 0.0;
	int c = 0;
	vector<double> meanlist;
	meanlist.resize(m_memory->m_N);
	for (int i = m_memory->m_N/2-1; i < m_memory->m_N; i++,c++){
		cv::Mat mean, stad;
		cv::meanStdDev(m_memory->m_imagesGray[i], mean, stad);
		//printf("%d %f %f\n", i, mean.at<double>(0, 0), stad.at<double>(0, 0));
		meanlist[i] = mean.at<double>(0, 0);
		avg += mean.at<double>(0, 0);
	}
	avg /= c; 
	
	for (int i = m_memory->m_N / 2-1; i < m_memory->m_N; i++){
		if (meanlist[i] < 0.05*avg){//0.85*avg
			//printf("%d\n", i);
			m_memory->m_imagesGray[i].setTo(0);
		}
	}
}

void MultiFusion::Saturation(){
	
	for (int i = 0; i < m_memory->m_N; i++){

		split(m_memory->m_images[i], m_memory->m_channels);
		cv::Mat R = m_memory->m_channels[2];
		cv::Mat G = m_memory->m_channels[1];
		cv::Mat B = m_memory->m_channels[0];

		m_memory->m_weight_saturat[i] = R + G;
		m_memory->m_weight_saturat[i] = m_memory->m_weight_saturat[i] + B;
		m_memory->m_weight_saturat[i] = m_memory->m_weight_saturat[i] / 3;
		
		R = R - m_memory->m_weight_saturat[i];
		G = G - m_memory->m_weight_saturat[i];
		B = B - m_memory->m_weight_saturat[i];

		pow(R, 2, R);
		pow(G, 2, G);
		pow(B, 2, B);

		m_memory->m_weight_saturat[i] = R + G;
		m_memory->m_weight_saturat[i] = m_memory->m_weight_saturat[i] + B;
		m_memory->m_weight_saturat[i] = m_memory->m_weight_saturat[i] / 3;
		
		sqrt(m_memory->m_weight_saturat[i], m_memory->m_weight_saturat[i]);
	}
}

void MultiFusion::WellExposedness(){
	
	for (int i = 0; i < m_memory->m_N; i++){

		split(m_memory->m_images[i], m_memory->m_channels);
		cv::Mat R = m_memory->m_channels[2];
		cv::Mat G = m_memory->m_channels[1];
		cv::Mat B = m_memory->m_channels[0];

		R = R - 0.5;
		R = R.mul(R);
		R = -12.5 * R;
		exp(R, R);

		G = G - 0.5;
		G = G.mul(G);
		G = -12.5 * G;
		exp(G, G);

		B = B - 0.5;
		B = B.mul(B);
		B = -12.5 * B;
		exp(B, B);

		R = R.mul(G);
		m_memory->m_weight_expose[i] = R.mul(B);
	}
}

void MultiFusion::MakeWeights(){
	int N = m_memory->m_N;
	for (int i = 0; i < N; i++)
	{
		m_memory->m_imagesGray[i] = m_memory->m_imagesGray[i].mul( m_memory->m_weight_saturat[i]);
		m_memory->m_imagesGray[i] = m_memory->m_imagesGray[i].mul( m_memory->m_weight_expose[i]);
		m_memory->m_imagesGray[i] = m_memory->m_imagesGray[i] + 1e-20;
		if (i == 0)  m_memory->m_weight_expose[0].setTo(0);
		m_memory->m_weight_expose[0] += m_memory->m_imagesGray[i];
	}
	for (int i = 0; i < N; i++)
		m_memory->m_imagesGray[i] /= m_memory->m_weight_expose[0];
}

void MultiFusion::MakeWeights(vector<cv::Mat> &masks){
	int N = m_memory->m_N;

	for (int i = 0; i < N; i++){
		masks[i].convertTo(masks[i],CV_32FC1);
		m_memory->m_imagesGray[i] = m_memory->m_imagesGray[i].mul(m_memory->m_weight_saturat[i]);
		m_memory->m_imagesGray[i] = m_memory->m_imagesGray[i].mul(m_memory->m_weight_expose[i]);
		m_memory->m_imagesGray[i] = m_memory->m_imagesGray[i].mul(masks[i]);
		m_memory->m_imagesGray[i] = m_memory->m_imagesGray[i] + 1e-12;
	
		if (i == 0)  m_memory->m_weight_expose[0].setTo(0);
		m_memory->m_weight_expose[0] += m_memory->m_imagesGray[i];
	}
	 
	for (int i = 0; i < N; i++)
		m_memory->m_imagesGray[i] /= m_memory->m_weight_expose[0];
}

void MultiFusion::reconstruct_laplacian_pyramid(){
	int r = m_memory->m_r;
	int c = m_memory->m_c;
	int nlev = m_memory->m_nlev;

	vector<cv::Size> list;
	list.push_back(cv::Size(c, r));
	for (int i = 0; i < nlev - 1; i++){
		double rr = 1.0*r;
		double cc = 1.0*c;
		rr /= 2;
		cc /= 2;
		r = int(rr + .5);
		c = int(cc + .5);
		list.push_back(cv::Size(c, r));
	}

	m_result = m_memory->m_Pyr[nlev - 1];
	for (int i = nlev - 2; i >= 0; i--){
		m_memory->upsample(m_result, m_memory->m_pyrI_auxiliary[i]);
		m_result = m_memory->m_Pyr[i] + m_memory->m_pyrI_auxiliary[i];
	}
}

//void MultiFusion::Execute(){
//
//	printf("Calc Fusion Weights...");
//	Contrast();  
//	Saturation();   
//	WellExposedness();  
//	MakeWeights();
//	//MakeWeights(masks);
//	printf("[DONE]\n");
//	printf("Merging...");
//	m_memory->m_DIYweightimages = m_memory->m_imagesGray;
//	//reconstruct according to Weights
//	for (int idx = 0; idx < m_memory->m_N; idx++){
//		m_memory->gaussian_pyramid(m_memory->m_imagesGray[idx], m_memory->m_pyrW); //m_pyrW现在存放GOP第idx张图片的权重的高斯金字塔
//		m_memory->laplacian_pyramid(m_memory->m_images[idx]);  //m_pyrI现在存放GOP第idx张原图片的拉普拉斯金字塔
//		printf("%02d\b\b", idx);
//		
//		for (int i = 0; i < m_memory->m_nlev; i++){
//			cv::Mat w = m_memory->repmatchannelx3(m_memory->m_pyrW[i]); 
//			m_memory->m_Pyr[i] += m_memory->m_pyrI[i].mul(w); 
//		}
//	}
//	printf("[DONE]\n");
//	
//	reconstruct_laplacian_pyramid();  
//}
void MultiFusion::Execute(vector<cv::Mat> &masks){

	//printf("Calc Fusion Weights...");
	//Contrast();
    //Saturation();
	//WellExposedness();
  //	MakeWeights();
	//MakeWeights(masks);
	//printf("[DONE]\n");
	printf("\nMerging...");
	//m_memory->m_DIYweightimages = m_memory->m_imagesGray;
	m_memory->m_DIYweightimages = masks;
	//reconstruct according to Weights
	
	for (int idx = 0; idx < m_memory->m_N; idx++){
		m_memory->gaussian_pyramid(m_memory->m_DIYweightimages[idx], m_memory->m_pyrW); //m_pyrW现在存放GOP第idx张图片的权重的高斯金字塔
		m_memory->laplacian_pyramid(m_memory->m_images[idx]);  //m_pyrI现在存放GOP第idx张原图片的拉普拉斯金字塔
		printf("%02d\b\b", idx);

		for (int i = 0; i < m_memory->m_nlev; i++){
			cv::Mat w = m_memory->repmatchannelx3(m_memory->m_pyrW[i]);
			m_memory->m_Pyr[i] += m_memory->m_pyrI[i].mul(w);
		}
	}
	printf("[DONE]\n");

	reconstruct_laplacian_pyramid();
}
MultiFusion::~MultiFusion(){
	delete m_memory;
}