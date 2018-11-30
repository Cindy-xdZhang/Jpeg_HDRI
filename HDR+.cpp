#include "HDR.h"
#define RESIZE 0//是否读入时统一化输入图片的尺寸
char * IMAGE_SEARCH_PATH = "inputimage/8/*.bmp";//字符串里注意不要出现空格！！！
char * JPEGIMAGE_SEARCH_PATH = "inputimage/8/result0/*.bmp";//ReadJPEGWeight函数读取jpeg编码信息bit图时的搜索路径
string IMAGE_OUTPUT_PATH = "outputimage/";
class JpegOperator{
public:
	vector<Mat>JpegBitsImages;
	void GenerateQtable(double quality = 50){
		if (quality <= 0)quality = 1;
		if (quality > 100)quality = 100;
		if (quality < 50)quality = 5000 / quality;//100+~5000
		else quality = 200 - quality * 2;//0~100
		double t[8][8] =
		{ 16, 11, 10, 16, 24, 40, 51, 61,\
		12, 12, 14, 19, 26, 58, 60, 55, \
		14, 13, 16, 24, 40, 57, 69, 56, \
		14, 17, 22, 29, 51, 87, 80, 62, \
		18, 22, 37, 56, 68, 109, 103, 77, \
		24, 35, 55, 64, 81, 104, 113, 92, \
		49, 64, 78, 87, 103, 121, 120, 101, \
		72, 92, 95, 98, 112, 100, 103, 99\
		};
		Qtable = Mat::zeros(8, 8, CV_64FC1);
		for (int i = 0; i < 8; ++i)
			for (int j = 0; j < 8; ++j)
			{
			t[i][j] = floor((t[i][j] * quality + 50) / 100.0);
			t[i][j];
			if (t[i][j] < 1)t[i][j] = 1;
			if (t[i][j] > 255)t[i][j] = 255;
			Qtable.at<double>(i, j) = t[i][j];
			}
		//Qtable.convertTo(Qtable,CV_8UC1,1,0);

	}
	//计算单张的 bit weight
	void Coding_Lum(Mat &src, Mat &dst, double quality = 50){
	//-------------------------转换色度空间rgb2yuv-----------
		/*Mat YUV;
		cv::cvtColor(src, YUV, CV_BGR2YUV);
		vector<Mat>mv;
		split(YUV, mv);
		Mat Y = mv[0].clone();
		mv.clear();
		Y.convertTo(Y, CV_64FC1, 1.0, 16);*/
		/*结果一直不太对*/
		//********************仔细对比发现MATLAB的对于位置灰度是比我高16故不再使用opencv库函数改为手写：YUV:Y=0.299*R+0.587*G+0.114*B  另外注意Rect( 7,  454, 4, 4)代表454行7列
		//YCbCr:Y=16+(65.738R+129.057G+25.064B)/256
	     vector<Mat>chanels;
		 split(src, chanels);
		Mat Y = Mat(src.size(), CV_64FC1);
		Y = (double)0.2567890625*chanels[2] + (double)0.50412890625*chanels[1] + (double)0.09790625*chanels[0] + 16;
		chanels.clear();
		Y.convertTo(Y, CV_64FC1, 1.0, 0);
		//Y经检验无误，type ：64FC1 224行352列
		GenerateQtable(quality);	//Q量化表无误
		int rows = src.rows;
		int cols = src.cols;
		int num1 = rows / 8;
		int num2 = cols / 8;
		vector<Mat>Ori_Blocks;
		vector<Mat>Dequan_Blocks;
		vector<Mat>Rec_dct_Blocks;
		vector<int>Bits_block;
		Mat dct_8x8_Block(8, 8, CV_64FC1);
		Mat Quan_Blocks_8x8(8, 8, CV_64FC1);
		Mat DeQuan_Block_8x8(8, 8, CV_64FC1);
		Mat Rec_Blocks_dct_8x8(8, 8, CV_64FC1);
		Mat Res_Blocks_8x8(8, 8, CV_64FC1);
		for (int j = 0; j < num2; j++)
		 for (int i = 0; i < num1; i++){
			cv::Mat tp2 = Y(cv::Rect(j * 8, i * 8, 8, 8));
			Ori_Blocks.push_back(tp2);
			}
		///initial完成 
		for (int k = 0; k < num1*num2; k++){
			if (k == 0){
				Ori_Blocks[k] = Ori_Blocks[k] - 128;

				cv::dct(Ori_Blocks[k], dct_8x8_Block);

				Quan_Blocks_8x8 = dct_8x8_Block / (Qtable);
				for (int i = 0; i < 8; i++)
					for (int j = 0; j < 8; j++)
					Quan_Blocks_8x8.at<double>(i, j) = cvRound(Quan_Blocks_8x8.at<double>(i, j));

				    DeQuan_Block_8x8 = Quan_Blocks_8x8.mul(Qtable);
					Dequan_Blocks.push_back(DeQuan_Block_8x8);

					Rec_Blocks_dct_8x8 = DeQuan_Block_8x8;
					Rec_dct_Blocks.push_back(Rec_Blocks_dct_8x8);

			
		
			}
			else{
				cv::dct(Ori_Blocks[k], dct_8x8_Block);
				Res_Blocks_8x8 = DC_prediction(Rec_dct_Blocks[k - 1], dct_8x8_Block);
				Quan_Blocks_8x8 = Res_Blocks_8x8 / (Qtable);
				for (int i = 0; i < 8; i++)
					for (int j = 0; j < 8; j++)
						Quan_Blocks_8x8.at<double>(i, j) = cvRound(Quan_Blocks_8x8.at<double>(i, j));

				DeQuan_Block_8x8 = Quan_Blocks_8x8.mul(Qtable);
				Dequan_Blocks.push_back(DeQuan_Block_8x8);

				Rec_Blocks_dct_8x8 = DC_reconstruction(Rec_dct_Blocks[k - 1], Dequan_Blocks[k]);
				Rec_dct_Blocks.push_back(Rec_Blocks_dct_8x8);
			//	if (k == 109) //证明现在codinglum没错了，问题可能出现在底层了
					//cout << Quan_Blocks_8x8 << endl;
			}
			int tpbit = Bitcount_Jpeg(Quan_Blocks_8x8);
			Bits_block.push_back(tpbit);
			//if (k == 209)//证明现在底层没错了，问题可能出现在截断
				//cout << tpbit << endl;
		}
		Mat Bit = Mat(Bits_block);
		Mat BITmap = Bit.reshape(0, num2);
		//大坑：matlab里图像分割是是列优先，即第二个8x8宏块是第二行第一列的宏块而不是第一行第二列，之前自己没注意到，c++直接习惯性的行优先分割，后来注意到了matlab这一点，把c++分割改成了列优先，\
		但没有改这里，后来才意识到，c++变成列优先分割后，这里的reshape相应的改成num2.其实可以c++完全采用行优先分割，这里直接num1，但是会和matlab列优先编码结果出现细微偏差，因此最后c++还是用列优先

		BITmap.convertTo(BITmap,CV_64FC1,1,0);
		double Max = 0; double Min = (double)INFINITY;
		for (int i = 0; i < BITmap.rows;i++)
		for (int j = 0; j < BITmap.cols; j++)
		{
			if (BITmap.at<double>(i, j)>Max)
				Max = BITmap.at<double>(i, j);
			if (BITmap.at<double>(i, j)<Min)
				Min = BITmap.at<double>(i, j);
		}
		BITmap = (BITmap - Min) / (Max - Min);
		BITmap= BITmap.t();
		dst = BITmap;
		////证明了和matlab一致
		//resize(BITmap, BITmap, cv::Size(), 8, 8);
		//Mat Ori = imread("inputimage/188/result0/002_bit1.bmp");
		//resize(Ori, Ori, cv::Size(), 8, 8);
		//	imshow("bit", BITmap); waitKey();
		//	imshow("bit2", Ori); waitKey();
	
	}
	//计算all bit weight
	void Jpeg_Sequence(vector<Mat>&images)
	{
		Mat tp;
		for (int i = 0; i < images.size(); i++)
		{
			this->Coding_Lum(images[i], tp);
			this->JpegBitsImages.push_back(tp);
		}

	}
private:
	Mat Qtable;
	Mat ACcount = (Mat_<uchar>(160, 1) << 2, 2, 3, 4, 5, 7, 8, 10, 16, 16, 4, 5, 7, 9, 11, 16, 16, 16, 16, 16, 5, 8, 10, 12, 16, 16, 16, 16, 16, 16, \
		6, 9, 12, 16, 16, 16, 16, 16, 16, 16, 6, 10, 16, 16, 16, 16, 16, 16, 16, 16, 7, 11, 16, 16, 16, 16, 16, 16, 16, 16, \
		7, 12, 16, 16, 16, 16, 16, 16, 16, 16, 8, 12, 16, 16, 16, 16, 16, 16, 16, 16, 9, 15, 16, 16, 16, 16, 16, 16, 16, 16, \
		9, 16, 16, 16, 16, 16, 16, 16, 16, 16, 9, 16, 16, 16, 16, 16, 16, 16, 16, 16, 10, 16, 16, 16, 16, 16, 16, 16, 16, 16, \
		10, 16, 16, 16, 16, 16, 16, 16, 16, 16, 11, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, \
		16, 16, 16, 16, 16, 16, 16, 16, 16, 16);
	//输入量化后的8x8块
	inline int Bitcount_Jpeg(Mat& Block){
		int Bits;
		if (sum(abs(Block))(0) != 0){
			vector<double>ZigZag;
			//==============按锯齿形的路径遍历8x8block==============================
			ZigZag.push_back(Block.at<double>(0, 0));
			ZigZag.push_back(Block.at<double>(0, 1)); ZigZag.push_back(Block.at<double>(1, 0));
			ZigZag.push_back(Block.at<double>(2, 0)); ZigZag.push_back(Block.at<double>(1, 1)); ZigZag.push_back(Block.at<double>(0, 2));
			ZigZag.push_back(Block.at<double>(0, 3)); ZigZag.push_back(Block.at<double>(1, 2)); ZigZag.push_back(Block.at<double>(2, 1)); ZigZag.push_back(Block.at<double>(3, 0));
			ZigZag.push_back(Block.at<double>(4, 0)); ZigZag.push_back(Block.at<double>(3, 1)); ZigZag.push_back(Block.at<double>(2, 2)); ZigZag.push_back(Block.at<double>(1, 3)); ZigZag.push_back(Block.at<double>(0, 4));
			ZigZag.push_back(Block.at<double>(0, 5)); ZigZag.push_back(Block.at<double>(1, 4)); ZigZag.push_back(Block.at<double>(2, 3)); ZigZag.push_back(Block.at<double>(3, 2)); ZigZag.push_back(Block.at<double>(4, 1)); ZigZag.push_back(Block.at<double>(5, 0));
			ZigZag.push_back(Block.at<double>(6, 0)); ZigZag.push_back(Block.at<double>(5, 1)); ZigZag.push_back(Block.at<double>(4, 2)); ZigZag.push_back(Block.at<double>(3, 3)); ZigZag.push_back(Block.at<double>(2, 4)); ZigZag.push_back(Block.at<double>(1, 5)); ZigZag.push_back(Block.at<double>(0, 6));
			ZigZag.push_back(Block.at<double>(0, 7)); ZigZag.push_back(Block.at<double>(1, 6)); ZigZag.push_back(Block.at<double>(2, 5)); ZigZag.push_back(Block.at<double>(3, 4)); ZigZag.push_back(Block.at<double>(4, 3)); ZigZag.push_back(Block.at<double>(5, 2)); ZigZag.push_back(Block.at<double>(6, 1)); ZigZag.push_back(Block.at<double>(7, 0));
			ZigZag.push_back(Block.at<double>(7, 1)); ZigZag.push_back(Block.at<double>(6, 2)); ZigZag.push_back(Block.at<double>(5, 3)); ZigZag.push_back(Block.at<double>(4, 4)); ZigZag.push_back(Block.at<double>(3, 5)); ZigZag.push_back(Block.at<double>(2, 6)); ZigZag.push_back(Block.at<double>(1, 7));
			ZigZag.push_back(Block.at<double>(2, 7)); ZigZag.push_back(Block.at<double>(3, 6)); ZigZag.push_back(Block.at<double>(4, 5)); ZigZag.push_back(Block.at<double>(5, 4)); ZigZag.push_back(Block.at<double>(6, 3)); ZigZag.push_back(Block.at<double>(7, 2));
			ZigZag.push_back(Block.at<double>(7, 3)); ZigZag.push_back(Block.at<double>(6, 4)); ZigZag.push_back(Block.at<double>(5, 5)); ZigZag.push_back(Block.at<double>(4, 6)); ZigZag.push_back(Block.at<double>(3, 7));
			ZigZag.push_back(Block.at<double>(4, 7)); ZigZag.push_back(Block.at<double>(5, 6)); ZigZag.push_back(Block.at<double>(6, 5)); ZigZag.push_back(Block.at<double>(7, 4));
			ZigZag.push_back(Block.at<double>(7, 5)); ZigZag.push_back(Block.at<double>(6, 6)); ZigZag.push_back(Block.at<double>(5, 7));
			ZigZag.push_back(Block.at<double>(6, 7)); ZigZag.push_back(Block.at<double>(7, 6));
			ZigZag.push_back(Block.at<double>(7, 7));
			Mat zigzag = Mat(ZigZag);//默认 zigzag是64x1的double列向量
			ZigZag.clear();
			//=============================SymbolFormation================================================		
			int I = 1;
			vector<double>Run;
			vector<double>Level;
			vector<Mat>Stack;
			int index = 0;
			Stack.push_back(zigzag);
			if (sum(abs(Stack[index]))(0) != 0)
				while (sum(abs(Stack[index]))(0) != 0){
				double SumResult = 0;
				double* Pdata = (double*)Stack[index].data;
				for (int i = 0; i< I; i++)
					SumResult += Pdata[i];
				while (SumResult == 0)
				{
					I += 1;
					for (int i = 0; i< I; i++)
						SumResult += Pdata[i];;

				}
				Run.push_back(I - 1);
				Level.push_back(1 + floor(log2(abs(Stack[index].at<double>(I - 1, 0)))));
				if (I <Stack[index].rows){
					Mat disseminate = Stack[index](cv::Rect(0, I, 1, Stack[index].rows - I));
					Stack.push_back(disseminate);
					index++;
					I = 1;
				}
				else break;
				}
			else{
				Run.push_back(0);
				Level.push_back(0);
			}
			Stack.clear();
			//============================================AC_bitcount==============================================================================
			int ACNumberofBits = 0;
			if (Run.size() != 1){
				for (int j = 1; j <Run.size(); j++){
					while (Run[j] >= 15){
						Run[j] -= 15;
						ACNumberofBits += 11;
					}
					int index = 10 * Run[j] + Level[j];
					if (index > 0)
						ACNumberofBits += ACcount.at<uchar>(index - 1, 0);
				}
			}
			ACNumberofBits += 4;
			//============================================DC_bitcount==============================================================================
			Mat count = (Mat_<uchar>(12, 1) << 2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9);
			int DC_Bit;
			if (zigzag.at<double>(0, 0) != 0) {
				int size = floor(log2(abs(zigzag.at<double>(0, 0)))) + 1;
				DC_Bit = count.at<uchar>(size, 0);
			}
			else
				DC_Bit = 2;//count(0)
			//==============================================================================================================================================
			Bits = DC_Bit + ACNumberofBits;
		}
		//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
		else Bits = 0;
		if (Bits < 0)cout << "Fatal Error in the countbit!" << endl;
		return Bits;
	}
	//经检查这两个函数无误，但必须工作在double模式
	inline  Mat DC_prediction(Mat&REF, Mat&CUR)
	{
		Mat res = CUR.clone();
		res.at<double>(0, 0) = CUR.at<double>(0, 0) - REF.at<double>(0, 0);
		return res;
	}
	inline  Mat DC_reconstruction(Mat&REF, Mat&RES)
	{
		Mat rec = RES.clone();
		rec.at<double>(0, 0) =  REF.at<double>(0, 0)+RES.at<double>(0, 0) ;
		return rec;
	}
};
class FrameOperator{
private:
	MultiFusion* FusionOprtor; //HDRI操作类
	int Framenum;//frame数目
	CvSize Framesize;//图片的长宽
	CvSize JPEGSIZE;
	vector<Mat>GOPWeightMat;//进入高斯金字塔的权重
	vector<Mat> JPEGWeights;
public:
	Mat HDRresult;
	vector<Mat>srcImages;//输入图片的图组
public:
    //构造函数
	FrameOperator(vector<Mat>images) 
	{
		this->Framesize = images[0].size();
		this->GOPWeightMat.reserve(images.size() + 5);
		this->srcImages = images;
		this->Framenum=images.size();
	}

	//使用opencv库的MERTENS方法进行HDRI合成
	void MertensHDRI(vector<Mat>&images){
		if (images.size() == 0)
		{
			cout << "NO Align Frames! Press any key back to the main function.." << endl;
			system("pause");
			return;
		}
		MergeMertens_HDRI(images);
	}
	
	//进行依赖于jpeg信息等的HDRI合成
	void DIYMultiFusion( vector<Mat>&imgs)
	{
		if (!imgs.size())
		{

			cout << " DIYMultiFusion function get no arguments ! Now return to main!" << endl;
			return ;
		}
		
        printf("MBProcessing....");
		for (int i = 0; i < imgs.size(); i++)
		{
			printf("%02d\b\b", i);
			JpegProcess(i);
		}
		if (GOPWeightMat.size() != imgs.size() )
		{
			cout << "Unknown Fatal Error In Jpegprocess!" << endl;
			cout << "Press a key to exit !" << endl;
			system("pause"); 
			exit(1);
		}
		BitChangeWeight();
		Uniformized();
		FusionOprtor = new MultiFusion(imgs);
		FusionOprtor->Execute(GOPWeightMat);
		this->HDRresult = FusionOprtor->GetResult();
		this->HDRresult.convertTo(this->HDRresult, CV_8UC3, 255, 0);

		delete FusionOprtor;
		this->JPEGWeights.clear();
		this->GOPWeightMat.clear();
		//this->Homographys.clear();
	}
	//为JPEG编码服务：jpeg编码前提：图片的行列为16的倍数
	void ResizeImageAndWrite(vector<Mat>&imgs)
	{	for (int i = 0; i < imgs.size(); i++)
		{
		float a, b; 
		string n;
			a = (float)1080/ imgs[i].cols;
			b = (float)1280/ imgs[i].rows;
			resize(imgs[i], imgs[i], cv::Size(), a, b);//此处resize warp后的图像对齐出现无法接受的误差
			n = to_string(i+1);
			char pDest[30];
			memcpy(pDest, IMAGE_SEARCH_PATH, strlen(IMAGE_SEARCH_PATH) - 5);
			pDest[strlen(IMAGE_SEARCH_PATH) - 5] = '\0';
			string IMAGE_OUTPREFIX_PATH = pDest;
			//imwrite( IMAGE_OUTPUT_PATH +"00"+n+".bmp", imgs[i]);
			imwrite(IMAGE_OUTPREFIX_PATH + "00" + n + ".bmp", imgs[i]);
		}
	cout << "Finish  " << imgs.size() << " pics Resizing !" << endl;
	}
private: 
	

//自己c++计算jpeg——bit权重的底层
void JpegProcess(int n)
{
	if (!this->JPEGWeights.size())
	{
		JpegOperator* jpS = new JpegOperator();
		jpS->Jpeg_Sequence(srcImages);
		for (int i = 0; i < jpS->JpegBitsImages.size(); i++)
			this->JPEGWeights.push_back(jpS->JpegBitsImages[i]);
		delete jpS;
	}


	if (this->JPEGWeights[n].rows * 8 != Framesize.height || JPEGWeights[n].cols * 8 != Framesize.width)
	{
		cout << endl << "Fatal Error in JpegProcess : JPEG size and Frame size not match!" << endl << " JPEG: " << this->JPEGWeights[n].size() << " Framesize: " << this->Framesize.width << " " << this->Framesize.height << endl;
		system("pause");
		exit(1);
	}
	if (this->JPEGWeights.size() != this->Framenum)
	{
		cout << endl << "Fatal Error in JpegProcess : JPEG num and Frame num not match!" << endl << " JPEG: " << JPEGWeights.size() << " Framenum: " << Framenum << endl;
		system("pause");
		exit(1);
	}
	//this->JPEGWeights[n].convertTo(this->JPEGWeights[n], CV_8UC1, 255, 0);
	//this->JPEGWeights[n].convertTo(this->JPEGWeights[n], CV_64FC1,1/255.0, 0);
	resize(this->JPEGWeights[n], this->JPEGWeights[n], cv::Size(), 8, 8);
	//cout << JPEGWeights[n].channels() << endl;
	//检测矩阵类型变换是否存在数据截断
	this->JPEGWeights[n].convertTo(JPEGWeights[n], CV_32FC1, 1.0, 0);
		this->GOPWeightMat.push_back(this->JPEGWeights[n]);//高斯金字塔输入必须权重必须为float型		

}
//使用opencv底层库进行MERTENS方法的HDRI合成
void MergeMertens_HDRI(vector<Mat>&images)
	{
	
		Mat fusion;
		Ptr<MergeMertens> merge_mertens = createMergeMertens();
		merge_mertens->process(images, fusion);
		fusion.convertTo(fusion, CV_8UC3, 255);
		HDRresult = fusion;
		//imwrite("hdri1.bmp", fusion);
	}	
//归一hdri化权重图
void Uniformized()
{
	int N = GOPWeightMat.size();
	Mat tpweight = Mat(GOPWeightMat[1].size(), GOPWeightMat[1].type());
	tpweight.setTo(0);
	for (int i = 0; i < N; i++){
		tpweight += GOPWeightMat[i];
	}
	for (int i = 0; i < N; i++)
		GOPWeightMat[i] /= tpweight;
}
//读取jpeg编码信息bit图
void ReadJPEGWeight()
{
	struct _finddata_t fileinfo;
	long long handle;//win7用long win10用longlong
	handle = _findfirst(JPEGIMAGE_SEARCH_PATH, &fileinfo);
	if (handle == -1){
		cout << " Funtion ReadJPEGWeight  fail to read pictures? right path?.." << endl;
		system("pause");
		exit(1);
	}
	int k = 0;
	Mat img;
	char pDest[30];
	memcpy(pDest, JPEGIMAGE_SEARCH_PATH, strlen(JPEGIMAGE_SEARCH_PATH) - 5);
	pDest[strlen(JPEGIMAGE_SEARCH_PATH) - 5] = '\0';
	string IMAGE_PREFIX_PATH = pDest;
	while (k != -1)
	{
		img = imread(IMAGE_PREFIX_PATH + fileinfo.name);
		if (RESIZE == 1)
		{
			float a, b;
			a = (float)1280 / img.cols;
			b = (float)1024 / img.rows;
			resize(img, img, cv::Size(), a, b);//此处resize warp后的图像对齐出现无法接受的误差

		}
		k = _findnext(handle, &fileinfo);
		// imshow("0", img); waitKey(0);
		 JPEGWeights.push_back(img);
	}
	JPEGSIZE = img.size();
	//cout << "JPEGWeights_imgsize" << img.size() << endl;
	_findclose(handle);
}
//由bit信息的调整权重
void BitChangeWeight()
{

	vector<float>Exposionvalue;//bit多少
	vector<float>UniformAreaBits;//平整区域bit多少（用来做补的平整区域权重）
	Mat uniform = Mat::ones(Framesize, CV_32FC1);
	/// ---------------------寻找平整区域（uniform）和曝光不对的区域badexposure
	for (int m = 0; m < GOPWeightMat.size(); m++)
	{
		for (int i = 0; i < Framesize.height; i++)
			for (int j = 0; j < Framesize.width; j++)
			{
			if (GOPWeightMat[m].at<float>(i, j) >= 0.08)
				uniform.at<float>(i, j) -= ((float)5 / (float)GOPWeightMat.size());
			if (uniform.at<float>(i, j) <= 0)
				uniform.at<float>(i, j) = 1e-12;
			}
	}
	//========================================================================================
	for (int i = 0; i < GOPWeightMat.size(); i++)
	{
		Exposionvalue.push_back(sum(GOPWeightMat[i])(0));
		UniformAreaBits.push_back(sum(GOPWeightMat[i].mul(uniform))(0));
	}
	double max1 = (double)0.0;
	double max2 = (double)0.0;
	double min = (double)INFINITY;
	for (int i = 0; i < GOPWeightMat.size(); i++)
	{
		if (UniformAreaBits[i]>max2)
			max2 = UniformAreaBits[i];
		if (Exposionvalue[i]>max1)
			max1= Exposionvalue[i];
	}
	for (int i = 0; i < GOPWeightMat.size(); i++)//弥补平整和调节色调
	{
		float weight2;
		weight2 = 1 - abs(max1 - Exposionvalue[i]) / max1;
		float weight3;
		weight3 = 1 - abs(max2 - UniformAreaBits[i]) / max2;
		weight3 = pow(weight3, 2);
	   GOPWeightMat[i] += (weight3*uniform);
	   GOPWeightMat[i] *= (weight2);//试验了几十天最终吧还是加上了weight2做权重
	}
	for (int i = 0; i < GOPWeightMat.size(); i++){
		string outname = ("inputimage/99/compensate_weight" + toString<int>(i)); outname += ".jpg";
		imwrite(outname, GOPWeightMat[i]*255);
	}

}
};


void ReadSequenceImage(vector<Mat>&images)
{
	struct _finddata_t fileinfo;
	long long handle;//win7用long win10用longlong
	handle = _findfirst(IMAGE_SEARCH_PATH, &fileinfo);
	if (handle == -1){
		cout << " Funtion ReadSequenceImage  fail to read pictures? right path?.." << endl;
		system("pause");
		exit(1);
	}
	int k = 0;
	Mat img;
	char pDest[30];
	memcpy(pDest, IMAGE_SEARCH_PATH, strlen(IMAGE_SEARCH_PATH) - 5);
	pDest[strlen(IMAGE_SEARCH_PATH) - 5] = '\0';
	string IMAGE_PREFIX_PATH = pDest;
	while (k != -1)
	{
		img = imread(IMAGE_PREFIX_PATH + fileinfo.name);
		if (RESIZE == 1)
		{
			float a, b;
			a = (float)1280 / img.cols;
			b = (float)1024/ img.rows;
			resize(img, img, cv::Size(), a, b);//此处resize warp后的图像对齐出现无法接受的误差

		}
		k = _findnext(handle, &fileinfo);
		images.push_back(img);
	}
	if (img.size().height == 0 || img.size().width == 0)
	{
		cout << "Read in pics  falied?!" << endl;
		cout << "imgsize:" << img.size() << "  " << "photo nums: " << images.size() << endl;
		system("pause");
		exit(1);
	}
	cout << "imgsize:" << img.size() <<"  "<< "photo nums: " << images.size() << endl;
	_findclose(handle);
}
/* Pstack = Pdisseminate;//直接zigzag=zigzag（少了列数的）无法成功，\
借助第三者mat传递也不行，mat.clone、mat.copyto都不行、试图利用指针把一个大的mat赋给一个小的mat，opencv报错，\
从某种角度证明opencv本身也是用指针进行操作的。最后的方法：vector */
void Demo(){
	IMAGE_SEARCH_PATH = "inputimage/99/*.bmp";
	string result_path="inputimage/99/result_out.jpg";
	vector<Mat>images;
	ReadSequenceImage(images);
	FrameOperator *FrameOprtor = new  FrameOperator(images);
	FrameOprtor->DIYMultiFusion(images);
	imwrite(result_path, FrameOprtor->HDRresult);
}
int main() 
{
	Demo();
	return 0;
}

