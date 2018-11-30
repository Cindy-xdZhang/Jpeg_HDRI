Mat Eresult;
int alpha_slider;
const int alpha_slider_max = 100;
class FrameOperator{
	struct Block
	{
		Point2i  mv;
		Point2i pt;
		uchar ref;
		Point2i MBpt;
	};
	struct MacroBlock
	{
		vector<Block> blk;
		Point2i pt;
		double weight;
	};
	struct TXT
	{
		vector<Block> BLK;
		int index;
	};

private:
	MultiFusion* FusionOprtor; //HDRI������
	int blknum;//blk��Ŀ
	int Framenum;//frame��Ŀ
	CvSize Framesize;//ͼƬ�ĳ���
	int FPS;//�ϳ���Ƶ��FPS	
	CvSize JPEGSIZE;
	vector<Mat>GOPWeightMat;//�����˹��������Ȩ��
	vector<Mat> JPEGWeights;
	//vector<cv::Mat> Homographys;
	//vector<TXT> TXTLIST;//txt�ļ���
	bool HomographyReadyFlag=false;
	
public:
	Mat HDRresult;
	uchar hdriflag = 0;//1:�����п⺯����2��������diyhdr,3:������seamcarvingbased_hdri,4:hdr+saturationAd
	vector<Mat>srcImages;//����ͼƬ��ͼ��
public:
	//ʹ��opencv�ײ��panoʽƴ��ͼƬ
	void BasicStitcher(const vector<Mat>&imgs)
	{
		Mat pano;//ƴ�ӽ��ͼƬ
		Stitcher stitcher = Stitcher::createDefault(true);//ʹ��GPU
		cout << "stitcher...." << endl;
		Stitcher::Status status = stitcher.stitch(imgs, pano);

		if (status != Stitcher::OK)
		{
			cout << "Can't stitch images, error code = " << int(status) << endl;
			system("pause");
			exit(1);
		}
		cout << "Finish..." << endl;
		imshow("pano", pano); waitKey(0);
	}
    //���캯��
	FrameOperator(vector<Mat>images) 
	{
		this->blknum = 0;//blk��
		this->FPS = 10;
		this->Framesize = images[0].size();
		/*this->TXTLIST.reserve(images.size() + 30);
		this->Homographys.reserve(images.size() + 5);*/
		this->GOPWeightMat.reserve(images.size() + 5);
		this->srcImages = images;
		this->Framenum=images.size();
	}
	~FrameOperator()
	{

	}
	
	//ʹ��opencv���MERTENS��������HDRI�ϳ�
	void MertensHDRI(vector<Mat>&images){
		if (images.size() == 0)
		{
			cout << "NO Align Frames! Press any key back to the main function.." << endl;
			system("pause");
			return;
		}this->hdriflag = 1;
		MergeMertens_HDRI(images);
	}
	
	//����������jpeg��Ϣ�ȵ�HDRI�ϳ�
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
	    this->hdriflag = 2;
		delete FusionOprtor;
		this->JPEGWeights.clear();
		this->GOPWeightMat.clear();
		//this->Homographys.clear();
	}
	//ΪJPEG�������jpeg����ǰ�᣺ͼƬ������Ϊ16�ı���
	void ResizeImageAndWrite(vector<Mat>&imgs)
	{	for (int i = 0; i < imgs.size(); i++)
		{
		float a, b; 
		string n;
			a = (float)1080/ imgs[i].cols;
			b = (float)1280/ imgs[i].rows;
			resize(imgs[i], imgs[i], cv::Size(), a, b);//�˴�resize warp���ͼ���������޷����ܵ����
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
	void WriteCurrentHDR_result()
	{	vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);  //ѡ��jpeg
		compression_params.push_back(100); //�����������Ҫ��ͼƬ����
		if (hdriflag == 0)
		{
			cout << "U Havent process HDR ,cant write the result!" << endl;
			return;
		}
		if (hdriflag == 1)
		{
			HDRresult.convertTo(HDRresult, CV_8UC3, 1, 0);
			char pDest[30];
			memcpy(pDest, IMAGE_SEARCH_PATH, strlen(IMAGE_SEARCH_PATH) - 5);
			pDest[strlen(IMAGE_SEARCH_PATH) - 5] = '\0';
			string IMAGE_OUTPREFIX_PATH = pDest;
			imwrite(IMAGE_OUTPREFIX_PATH + "MERTENresult.jpg", HDRresult, compression_params);

		}
		if (hdriflag == 2)
		{
			char pDest[30];
			memcpy(pDest, IMAGE_SEARCH_PATH, strlen(IMAGE_SEARCH_PATH) - 5);
			pDest[strlen(IMAGE_SEARCH_PATH) - 5] = '\0';
			string IMAGE_OUTPREFIX_PATH = pDest;
			imwrite(IMAGE_OUTPREFIX_PATH + "MYDIYresult.jpg", HDRresult, compression_params);

		}
		if (hdriflag == 3)
		{
			char pDest[30];
			memcpy(pDest, IMAGE_SEARCH_PATH, strlen(IMAGE_SEARCH_PATH) - 5);
			pDest[strlen(IMAGE_SEARCH_PATH) - 5] = '\0';
			string IMAGE_OUTPREFIX_PATH = pDest;
			imwrite(IMAGE_OUTPREFIX_PATH + "SeamCarvingHDRresult.jpg", HDRresult, compression_params);

		}
		if (hdriflag == 4)
		{
			char pDest[30];
			memcpy(pDest, IMAGE_SEARCH_PATH, strlen(IMAGE_SEARCH_PATH) - 5);
			pDest[strlen(IMAGE_SEARCH_PATH) - 5] = '\0';
			string IMAGE_OUTPREFIX_PATH = pDest;
			imwrite(IMAGE_OUTPREFIX_PATH + "MYProcessResult.jpg", HDRresult, compression_params);

		}
		if (hdriflag == 5)
		{
			char pDest[30];
			memcpy(pDest, IMAGE_SEARCH_PATH, strlen(IMAGE_SEARCH_PATH) - 5);
			pDest[strlen(IMAGE_SEARCH_PATH) - 5] = '\0';
			string IMAGE_OUTPREFIX_PATH = pDest;
			imwrite(IMAGE_OUTPREFIX_PATH + "C++_JpResult.jpg", HDRresult, compression_params);

		}
		cout << "WRITE SUCCESSFULLY!" << endl;
	}

private: 
	

//�Լ�c++����jpeg����bitȨ�صĵײ�
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
	//���������ͱ任�Ƿ�������ݽض�
	this->JPEGWeights[n].convertTo(JPEGWeights[n], CV_32FC1, 1.0, 0);
		this->GOPWeightMat.push_back(this->JPEGWeights[n]);//��˹�������������Ȩ�ر���Ϊfloat��		

}
//ʹ��opencv�ײ�����MERTENS������HDRI�ϳ�
void MergeMertens_HDRI(vector<Mat>&images)
	{
	
		Mat fusion;
		Ptr<MergeMertens> merge_mertens = createMergeMertens();
		merge_mertens->process(images, fusion);
		fusion.convertTo(fusion, CV_8UC3, 255);
		HDRresult = fusion;
		//imwrite("hdri1.bmp", fusion);
	}	
//��һhdri��Ȩ��ͼ
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
//��ȡjpeg������Ϣbitͼ
void ReadJPEGWeight()
{
	struct _finddata_t fileinfo;
	long long handle;//win7��long win10��longlong
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
			resize(img, img, cv::Size(), a, b);//�˴�resize warp���ͼ���������޷����ܵ����

		}
		k = _findnext(handle, &fileinfo);
		// imshow("0", img); waitKey(0);
		 JPEGWeights.push_back(img);
	}
	JPEGSIZE = img.size();
	//cout << "JPEGWeights_imgsize" << img.size() << endl;
	_findclose(handle);
}
//��bit��Ϣ�ĵ���Ȩ��
void BitChangeWeight()
{

	vector<float>Exposionvalue;//bit����
	vector<float>UniformAreaBits;//ƽ������bit���٣�����������ƽ������Ȩ�أ�
	Mat uniform = Mat::ones(Framesize, CV_32FC1);
	/// ---------------------Ѱ��ƽ������uniform�����عⲻ�Ե�����badexposure
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
	for (int i = 0; i < GOPWeightMat.size(); i++)//�ֲ�ƽ���͵���ɫ��
	{
		float weight2;
		weight2 = 1 - abs(max1 - Exposionvalue[i]) / max1;
		float weight3;
		weight3 = 1 - abs(max2 - UniformAreaBits[i]) / max2;
		weight3 = pow(weight3, 2);
	   GOPWeightMat[i] += (weight3*uniform);
	   GOPWeightMat[i] *= (weight2);//�����˼�ʮ�����հɻ��Ǽ�����weight2��Ȩ��
	}
	for (int i = 0; i < GOPWeightMat.size(); i++){
		string outname = ("inputimage/99/compensate_weight" + toString<int>(i)); outname += ".jpg";
		imwrite(outname, GOPWeightMat[i]*255);
	}

}
};
//���ļ������������õĴ���
void SequenceResize()
{

	vector<Mat>images;
	char str1[30];
	char str2[30];
	for (int m = 23; m <= 41; m++)
	{
		cout << "File Number: " << m << endl;
		sprintf(str1, "inputimage/%d/*.jpg", m);
		IMAGE_SEARCH_PATH = str1;
		ReadSequenceImage(images);
		FrameOperator* PF = new FrameOperator(images);
		PF->ResizeImageAndWrite(images);
		delete  PF;
		images.clear();

	}
}
//RGB�任HSL�ռ䣬���ߺϳɽ���ı��Ͷ�Ĭ�ϵ���20.0
void AdSaturation_HSL()
{
	cv::destroyAllWindows();
	Mat hdrresult;
	//hdrresult=imread("inputimage/13/MERTENresult.jpg" );
	this->HDRresult.copyTo(hdrresult);
	hdrresult.convertTo(hdrresult, CV_64FC3, 1.0, 0);
	int rows = hdrresult.rows;
	int cols = hdrresult.cols;
	vector<Mat>rgbChannels;//BGR
	Mat R_new, G_new, B_new;
	split(hdrresult, rgbChannels);
	R_new = rgbChannels[2];
	G_new = rgbChannels[1];
	B_new = rgbChannels[0];
	double increment = 20.0 / 100.0;
	//=======================================================
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
		double rgbMax = max(max(rgbChannels[0].at<double>(i, j), rgbChannels[2].at<double>(i, j)), rgbChannels[1].at<double>(i, j));
		double rgbMin = min(min(rgbChannels[0].at<double>(i, j), rgbChannels[2].at<double>(i, j)), rgbChannels[1].at<double>(i, j));
		double delta = (rgbMax - rgbMin) / 255;
		if (delta == 0)
			continue;
		double value = (rgbMax + rgbMin) / 255;
		double L = value / 2;
		double S;

		if (L <= 0.5)
			S = delta / value;
		else
			S = delta / (2.0 - value);
		double Alpha;
		if (increment >= 0)
		{
			if ((increment + S) >= 1)
				Alpha = S;
			else
				Alpha = 1 - increment;
			Alpha = 1 / Alpha - 1;
			R_new.at<double>(i, j) = rgbChannels[2].at<double>(i, j) + (rgbChannels[2].at<double>(i, j) - L * 255) * Alpha;
			G_new.at<double>(i, j) = rgbChannels[1].at<double>(i, j) + (rgbChannels[1].at<double>(i, j) - L * 255) * Alpha;
			B_new.at<double>(i, j) = rgbChannels[0].at<double>(i, j) + (rgbChannels[0].at<double>(i, j) - L * 255) * Alpha;
		}
		else
		{
			Alpha = increment;
			R_new.at<double>(i, j) = L * 255 + (rgbChannels[2].at<double>(i, j) - L * 255) * (1 + Alpha);
			G_new.at<double>(i, j) = L * 255 + (rgbChannels[1].at<double>(i, j) - L * 255) * (1 + Alpha);
			B_new.at<double>(i, j) = L * 255 + (rgbChannels[0].at<double>(i, j) - L * 255) * (1 + Alpha);
		}
		}
	Mat SaturatinAdResult;
	vector<Mat>newchanels;
	newchanels.push_back(B_new);
	newchanels.push_back(G_new);
	newchanels.push_back(R_new);
	cv::merge(newchanels, SaturatinAdResult);
	hdrresult.convertTo(hdrresult, CV_8UC3, 1, 0);
	SaturatinAdResult.convertTo(SaturatinAdResult, CV_8UC3, 1, 0);
	imshow("old", hdrresult);
	imshow("new", SaturatinAdResult);
	waitKey();
	this->HDRresult = SaturatinAdResult;
}
void on_trackbar(int, void*)
{
	Mat stack;
	Eresult.copyTo(stack);
	double alpha;
	alpha = (double)alpha_slider / alpha_slider_max;


	Eresult.convertTo(Eresult, CV_64FC3, 1.0, 0);
	int rows = Eresult.rows;
	int cols = Eresult.cols;
	vector<Mat>rgbChannels;//BGR
	Mat R_new, G_new, B_new;
	split(Eresult, rgbChannels);

	R_new = rgbChannels[2];
	G_new = rgbChannels[1];
	B_new = rgbChannels[0];
	double increment = (double)alpha;
	//=======================================================
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
		double rgbMax = max(max(rgbChannels[0].at<double>(i, j), rgbChannels[2].at<double>(i, j)), rgbChannels[1].at<double>(i, j));
		double rgbMin = min(min(rgbChannels[0].at<double>(i, j), rgbChannels[2].at<double>(i, j)), rgbChannels[1].at<double>(i, j));
		double delta = (rgbMax - rgbMin) / 255;
		if (delta == 0)
			continue;
		double value = (rgbMax + rgbMin) / 255;
		double L = value / 2;
		double S;

		if (L <= 0.5)
			S = delta / value;
		else
			S = delta / (2.0 - value);
		double Alpha;
		if (increment >= 0)
		{
			if ((increment + S) >= 1)
				Alpha = S;
			else
				Alpha = 1 - increment;
			Alpha = 1 / Alpha - 1;
			R_new.at<double>(i, j) = rgbChannels[2].at<double>(i, j) + (rgbChannels[2].at<double>(i, j) - L * 255) * Alpha;
			G_new.at<double>(i, j) = rgbChannels[1].at<double>(i, j) + (rgbChannels[1].at<double>(i, j) - L * 255) * Alpha;
			B_new.at<double>(i, j) = rgbChannels[0].at<double>(i, j) + (rgbChannels[0].at<double>(i, j) - L * 255) * Alpha;
		}
		else
		{
			Alpha = increment;
			R_new.at<double>(i, j) = L * 255 + (rgbChannels[2].at<double>(i, j) - L * 255) * (1 + Alpha);
			G_new.at<double>(i, j) = L * 255 + (rgbChannels[1].at<double>(i, j) - L * 255) * (1 + Alpha);
			B_new.at<double>(i, j) = L * 255 + (rgbChannels[0].at<double>(i, j) - L * 255) * (1 + Alpha);
		}
		}
	Mat SaturatinAdResult;
	vector<Mat>newchanels;
	newchanels.push_back(B_new);
	newchanels.push_back(G_new);
	newchanels.push_back(R_new);
	cv::merge(newchanels, SaturatinAdResult);
	SaturatinAdResult.convertTo(SaturatinAdResult, CV_8UC3, 1, 0);
	imshow("Linear Blend", SaturatinAdResult);
	stack.copyTo(Eresult);
}
void WindowUI_OP()
{
	if (Eresult.type() == 21)
		Eresult.convertTo(Eresult, CV_8UC3, 255, 0);
	namedWindow("Linear Blend", 1);
	/// �ڴ����Ĵ����д���һ���������ؼ�
	char TrackbarName[50];
	sprintf(TrackbarName, "Alpha x %d", alpha_slider_max);
	createTrackbar(TrackbarName, "Linear Blend", &alpha_slider, alpha_slider_max, on_trackbar);

	/// ����ڻص���������ʾ
	on_trackbar(alpha_slider, 0);

	/// ��������˳�
	waitKey(0);
}
void SequenceOperation()
{
	vector<Mat>images;
	char str1[30];
	char str2[30];
	for (int m = 5; m <=9; m++)
	{
		cout << "File Number: "<<m << endl;
		sprintf(str1, "inputimage/%d/*.bmp", m);
		IMAGE_SEARCH_PATH = str1;
		ReadSequenceImage(images);
		sprintf(str2, "inputimage/%d/result0/*.bmp", m);
		JPEGIMAGE_SEARCH_PATH = str2;
		FrameOperator* PF = new FrameOperator(images);
		//PF->ResizeImageAndWrite(images);
		PF->JpegDIYMultiFusion_AdSaturation_HSL(images);
		PF->WriteCurrentHDR_result();
		PF->DIYMultiFusion(images);
		PF->WriteCurrentHDR_result();
		//PF->MertensHDRI(images);
		//PF->WriteCurrentHDR_result();
		delete  PF;
		images.clear();
	}
	cout << "Finish All.." << endl;
	system("pause");
	exit(0);

}
void DIY_Seam_Carving_Based_MultiFusion(vector<Mat>&imgs)
{
	if (!imgs.size())
	{

		cout << " DIYMultiFusion function get no arguments ! Now return to main!" << endl;
		return;
	}

	printf("MBProcessing....");
	for (int i = 0; i < imgs.size(); i++)
	{
		printf("%02d\b\b", i); 
		Mat e = get_energy(imgs[i]);
		e.convertTo(e,CV_32FC1, 1, 0);
		GOPWeightMat.push_back(e);
		imshow("SeamCarvingEnergy", GOPWeightMat[i]);
		imshow("src", imgs[i]);
		waitKey(0);
	}
	if (GOPWeightMat.size() != imgs.size())
	{
		cout << "Unknown Fatal Error In MBprocess!" << endl;
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
	this->hdriflag = 3;
	delete FusionOprtor;
	this->JPEGWeights.clear();
	this->GOPWeightMat.clear();
	this->Homographys.clear();
	}
void JpegDIYMultiFusion_AdSaturation_HSL(vector<Mat>&imgs)
{
	if (!imgs.size())
	{

		cout << " DIYMultiFusion_AdSaturation_HSL function get no arguments ! Now return to main!" << endl;
		return;
	}

	printf("JpegProcessing....");
	for (int i = 0; i < imgs.size(); i++)
	{
		printf("%02d\b\b", i);
		JpegProcess(i);
	}
	printf("Finished!");
	if (GOPWeightMat.size() != imgs.size())
	{
		cout << "Unknown Fatal Error In MBprocess!" << endl;
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
	this->hdriflag = 5;
	delete FusionOprtor;
	this->JPEGWeights.clear();
	this->GOPWeightMat.clear();
	this->Homographys.clear();
	//imshow("C++result", this->HDRresult);
	//waitKey();
	//	AdSaturation_HSL();
}
//HDRȨ�ؼ���ϳ�+���ߺϳɽ���ı��Ͷ�+дͼƬ
void DIYMultiFusion_AdSaturation_HSL_Write(vector<Mat>&imgs)
{
	if (!imgs.size())
	{

		cout << " DIYMultiFusion_AdSaturation_HSL function get no arguments ! Now return to main!" << endl;
		return;
	}

	printf("MBProcessing....");
	for (int i = 0; i < imgs.size(); i++)
	{
		printf("%02d\b\b", i);
		MacroBlockProcess(i);
	}
	if (GOPWeightMat.size() != imgs.size())
	{
		cout << "Unknown Fatal Error In MBprocess!" << endl;
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
	this->hdriflag = 4;
	delete FusionOprtor;
	this->JPEGWeights.clear();
	this->GOPWeightMat.clear();
	this->Homographys.clear();
	AdSaturation_HSL();
	WriteCurrentHDR_result();
}
//ջ�����վ�ϵĺܱ��ܲ�ĵ����Ͷȷ�����ֻ�Ƽ����˽�����Ͷȵ�˼·���任�ռ䡢increment�����rgb�ռ䣩��ԭ������FrameOperator�ࣩ
void AdSaturation()
{
	cv::destroyAllWindows();
	Mat hdrresult;
	this->HDRresult.copyTo(hdrresult);

	const int alpha_slider_max = 100;
	int alpha_slider;
	// BGR to HSV
	cvtColor(hdrresult, hdrresult, CV_BGR2HSV);
	for (int i = 0; i < hdrresult.rows; i++)
	{
		for (int j = 0; j < hdrresult.cols; j++)
		{
			// You need to check this, but I think index 1 is for saturation, but it might be 0 or 2
			int idx = 1;
			//tp.at<cv::Vec3b>(i, j)[idx] = new_value;

			// or:
			hdrresult.at<cv::Vec3b>(i, j)[idx] += (uchar)50;
		}
	}
	// HSV back to BGR
	cvtColor(hdrresult, hdrresult, CV_HSV2BGR);
	imshow("Linear Blend", hdrresult);

	// ��������˳�
	waitKey(0);


}

//������ԭ������FrameOperator�ࣩ

//�ϳ���Ƶ
void Frames2Video(const vector<Mat>&imgs)
{
	if (imgs.size() == 0)
	{
		cout << "Error in Frames2Video :ThereThere s NO Input image! Press a key to exit." << endl;
		getchar();
		exit(0);
	}
	cout << "-----Video_To_Image------Framenum:" << imgs.size() << "  Framesize" << imgs[0].size() << endl;
	cv::VideoWriter writer;
	string video_name = "out1.avi";
	writer = VideoWriter(video_name, CV_FOURCC('M', 'J', 'P', 'G'), FPS,  imgs[0].size(), true);
	for (int i = 0; i < imgs.size(); i++)
	{
		//imshow("s", imgs[i]); waitKey(10);
		writer << imgs[i];
	}
	system("pause");
}
//����seamcarving Energy
Mat get_energy(const cv::Mat &srceimage)//srcΪ��ͨ��ԭ��ͼ8UC3�õ�һ��ת��Ϊ1/255��double�ε�ͨ���ݶ�����ͼ
{

	cv::Mat gradx, grady, abs_gradx, abs_grady, Energy, srcimage;
	cv::cvtColor(srceimage, srcimage, CV_BGR2GRAY);//�ǳ��ؼ�
	cv::Sobel(srcimage, grady, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(grady, abs_grady);
	cv::Sobel(srcimage, gradx, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(gradx, abs_gradx);
	addWeighted(abs_gradx, 0.5, abs_grady, 0.5, 0, Energy);
	Energy.convertTo(Energy, CV_64FC1, 1.0 / 255.0, 0);
	return Energy;
}
//��ȡtxt��mv��Ϣ
inline vector<Block> ReadTXT(string PREFIX_PATH)
{
	FILE *fp = fopen(PREFIX_PATH.c_str(), "r");
	if (fp == NULL)
	{
		cout << "Open TXT error ,Right Path?: " << PREFIX_PATH << endl;
		system("pause");
		exit(1);
	}
	vector<Block>srcblklist1;
	while (!feof(fp))
	{
		Block block;
		int pir;
		if (fscanf(fp, "block location:(%d,%d) Mb location:(%d,%d) mv%d:(%d, %d) refrence%d:%d", &block.pt.x, &block.pt.y, &block.MBpt.x, &block.MBpt.y, &pir, &block.mv.x, &block.mv.y, &pir, &block.ref) != EOF)
		{
			char ch = fgetc(fp);
			if ((ch == '\n'))
			{
				srcblklist1.push_back(block);

			}
			else //����B֡�����ⲿ��
			{
				fscanf(fp, "mv%d:(%d, %d) refrence%d : %d\n", &pir, &pir, &pir, &pir, &pir);
				srcblklist1.push_back(block);
			}
		}
	}
	return srcblklist1;
}
//������֡�ĵײ㺯��
inline void TwoframeAlign(const Mat&img1, const Mat&img2, Mat&img1to2, int n)
{
	char str[3];	
	string Txt_PATH;
	sprintf(str, "%d", n);
	string strtp(str);
	Txt_PATH = TXT_PATHPREFIX + strtp + ".txt";
	TXT txt; 
	txt.BLK = ReadTXT(Txt_PATH);
	txt.index = n;
	if (blknum == 0)
		blknum = txt.BLK.size();
	else 
		if (blknum != txt.BLK.size())
		{ 
		cout << "txt wrong cols!" << endl;
		system("pause");
		exit(1);
		}

	TXTLIST.push_back(txt);
	vector<Point2i>srclist2, srclist1,MV;
	Point2i block; 
	MV = CalMotionVector(n);	
	for (int i = 0; i < blknum; i++)
	{
		srclist1.push_back(txt.BLK[i].pt);
		block = MV[i] + txt.BLK[i].pt;
		srclist2.push_back(block);
	}
	cv::Mat h1 = findHomography(srclist1, srclist2, CV_RANSAC, 3);//�ж��Ƿ�Ϊ�ڵ��therohold=3������
	Homographys.push_back(h1);
	srclist1.clear();
	srclist2.clear();
	txt.BLK.clear();
	cv::Mat imageTransform1;
	warpPerspective(img1, imageTransform1, h1, cv::Size(img1.cols, img1.rows));
	img1to2 = imageTransform1;
	/*	imshow("��һ֡", img2);
	waitKey(0);
	imshow("ԭ֡", img1);
	waitKey(0);
	imshow("�任֡", imageTransform1);
	waitKey(0);*/
	}
inline vector<Block> ReadTXT(char * PREFIX_PATH)
{

	FILE *fp = fopen(PREFIX_PATH, "r");
	if (fp == NULL)
	{
		cout << "Open TXT error ,Right Path? ....." << endl;
		system("pause");
		exit(1);
	}
	vector<Block>srcblklist1;
	cout << "path:" << PREFIX_PATH << endl;
	while (!feof(fp))
	{

		Block block;
		int  mbx, mby, pir;
		if (fscanf(fp, "block location:(%d,%d) Mb location:(%d,%d) mv%d:(%d, %d) refrence%d:%d", &block.pt.x, &block.pt.y, &mbx, &mby, &pir, &block.mv.x, &block.mv.y, &pir, &block.ref) != EOF)
		{
			char ch = fgetc(fp);
			if ((ch == '\n'))
			{
				srcblklist1.push_back(block);

			}
			else //����B֡�����ⲿ��
			{
				int pir1, ref1;
				Point2i mv1;
				fscanf(fp, "mv%d:(%d, %d) refrence%d : %d\n", &pir1, &mv1.x, &mv1.y, &pir1, &ref1);
				srcblklist1.push_back(block);
			}
		}
	}
	return srcblklist1;
}
// ������n��txt���m��block��mv(��m+1��block)
inline Point2i CalOneMotionVector(int n, int m)
{
	int LISIindex = 0; bool LISTBUFF = false;
	for (int k = 0; k < TXTLIST.size(); k++)
	{
		if (TXTLIST[k].index == n)
		{
			LISTBUFF = true;
			LISIindex = k;
			break;
		}
	}
	Point2i block;
	if (LISTBUFF == true)
	{
		if (LISIindex >= TXTLIST.size())
		{
			cout << "FATAL ERROR, CODE12!" << endl;
			system("pause");
			exit(0);
		}
		for (int j = 1; j <= 16; j++)
		{
			if (TXTLIST[LISIindex].BLK[m].ref == 0)
			{
				block = TXTLIST[LISIindex].BLK[m].mv;
				break;
			}
			if (TXTLIST[LISIindex].BLK[m].ref == n - j)
				block = TXTLIST[LISIindex].BLK[m].mv + CalOneMotionVector(n - j, m);
			if (TXTLIST[LISIindex].BLK[m].ref == n + j)
				block = TXTLIST[LISIindex].BLK[m].mv + CalOneMotionVector(n + j, m);

		}
	}
	if (LISTBUFF == false)
	{
		char str[3];
		string Txt_PATH;
		sprintf(str, "%d", n);
		string strtp(str);
		Txt_PATH = TXT_PATHPREFIX + strtp + ".txt";
		TXT txt;
		txt.BLK = ReadTXT(Txt_PATH);
		txt.index = n;
		TXTLIST.push_back(txt);
		for (int j = 1; j <= 16; j++)
		{

			if (txt.BLK[m].ref == 0)
			{
				block = txt.BLK[m].mv;
				break;
			}
			if (txt.BLK[m].ref == n - j)
			{
				block = txt.BLK[m].mv + CalOneMotionVector(n - j, m);
			}
			if (txt.BLK[m].ref == n + j)
			{
				block = txt.BLK[m].mv + CalOneMotionVector(n + j, m);
			}
		}
	}
	return block;
}
//������n��txt��mv(���ڵ�һ֡)
inline vector<Point2i> CalMotionVector(int n)
{
	Point2i block;
	vector<Point2i> MV;
	for (int i = 0; i <blknum; ++i)
	{
		block = CalOneMotionVector(n, i);
		MV.push_back(block);
	}
	return MV;
}
//���һ֡�����֡
void FramesAlign(vector<Mat>images, vector<Mat>&HDRimages, int startindex, int endindex)//Ĭ�ϵ�һ֡(0)ΪIDR֡������֡����һ֡����
{
	if (HomographyReadyFlag)
	{
		cout << "U had Already Align the pics,the function cant work!" << endl;
		return;
	}
	if (images.size() < (endindex - startindex) + 1)
	{
		cout << "U get only " << images.size() << "pics,the function cant work!" << endl;
		getchar();
		exit(1);
	}
	vector<Mat>Result;
	Mat result;
	printf("\nAligning...");
	for (int n = 0; n < endindex - startindex; ++n)
	{
		printf("%02d\b\b", n);
		TwoframeAlign(images[n + 1], images[0], result, n + 1);
		Result.push_back(result);
	}

	HDRimages.push_back(images[0]);
	for (int i = 0; i < endindex; i++)
		HDRimages.push_back(Result[i]);
	TXTLIST.clear();
	if (Homographys.size() == endindex - startindex)HomographyReadyFlag = true;
}

void ConsoleWindowTestCircle()
{
	vector<Mat>images;
	ReadSequenceImage(images);
	bool now = true;
	vector<Mat>HDRimages;
	FrameOperator *FrameOprtor = new  FrameOperator(images);
	cout << "what are u going to do? \nps:Ĭ�Ͻ��е���������������sequenceOperation������p��������ǰ����inputimage�ļ�������.\nsequenceOperation��ɺ󽫲��ٻص�����̨��ֱ���˳�" << endl \
		<< "  PRESS a : �ϳ���Ƶ" << endl << "  PRESS b : REsizeͼƬ"\
		<< endl << "  PRESS c : DIYMultiFusion" << endl \
		<< "  PRESS d : MertensHDRI images" << endl << "  PRESS e : MertensHDRI alignimages" \
		<< endl << "  PRESS f :  Align images" << endl << "  PRESS g :  Show The Current src images " << \
		endl << "  PRESS i :  Show The Current src images Information " << endl << "  PRESS q :  Quit EXE.."\
		<< endl << "  PRESS h :  GET help information.." << endl << "  PRESS j:  DIYMultiFusion the aligns.." \
		<< endl << "  PRESS k :  Imshow and Write previous HDRI result.." \
		<< endl << "  PRESS l :  SeamCarving Based HDRI.." << endl << "  PRESS m :  MultiFusion_AdSaturation_HSL_Write.." << endl << "  PRESS n :  Get ASResult Window.." << endl\
		<< "  PRESS o :  c++ Based HDRI.." << endl;

	while (now)
	{
		char snode = getch();
		switch (snode)
		{
		case 97://a
		{
			FrameOprtor->Frames2Video(images);
			break;
		}
		case 98://b
		{
			FrameOprtor->ResizeImageAndWrite(images);
			break;
		}
		case 99://c
		{
			FrameOprtor->DIYMultiFusion(images);
			imshow("DIYhdri", FrameOprtor->HDRresult);
			waitKey(0);
			break;
		}
		case 100:
		{
			FrameOprtor->MertensHDRI(images);
			imshow("MertensHDRI", FrameOprtor->HDRresult);
			waitKey(0);
			break;
		}
		case 101:
		{	FrameOprtor->MertensHDRI(HDRimages);
		break;
		}
		case 102:
		{	FrameOprtor->FramesAlign(images, HDRimages, 0, images.size() - 1);
		break;
		}

		case 104://h
		{cout << "����������" << endl;
		cout << "  PRESS a : �ϳ���Ƶ" << endl << "  PRESS b : REsizeͼƬ"\
			<< endl << "  PRESS c : DIYMultiFusion" << endl \
			<< "  PRESS d : MertensHDRI images" << endl << "  PRESS e : MertensHDRI alignimages" \
			<< endl << "  PRESS f :  Align images" << endl << "  PRESS g :  Show The Current src images " << \
			endl << "  PRESS i :  Show The Current src images Information " << endl << "  PRESS q :  Quit EXE.."\
			<< endl << "  PRESS h :  GET help information.." << endl << "  PRESS j:  DIYMultiFusion the aligns.." << endl \
			<< "  PRESS k :  Imshow and Write previous HDRI result.." \
			<< endl << "  PRESS l :  SeamCarving Based HDRI.." << endl << "  PRESS m :  MultiFusion_AdSaturation_HSL_Write.." << endl << "  PRESS n :  Get ASResult Window.." << endl
			<< "  PRESS o :  c++ Based HDRI.." << endl;
		cout << "ע�����" << endl;
		cout << "1. �Զ���ͼƬ����hdri�ϳ�֮ǰ����ʹ��f��Align images�õ������ͼƬ" << endl;
		cout << "2. ʹ��  Align images��DIYMultiFusionӦע�������ļ�Ŀ¼��txt jpeg Ŀ¼�Ƿ�match" << endl;
		cout << "3.Ĭ�Ͻ��е���������������sequenceOperation������p��������ǰ����inputimage�ļ�������" << endl;
		break;
		}
		case 103://g
		{
			for (int i = 0; i < images.size(); i++)
			{
				imshow("Current Working Images", images[i]);
				waitKey(0);
			}
			break;
		}
		case 105://i
		{   cout << " FRAME NUM:" << images.size() << endl;
		cout << " FRAME SIZE:" << images[1].size() << endl;
		cout << " FRAME channels:" << images[1].channels() << endl;
		cout << " FRAME type:" << images[1].type() << endl;
		break;
		}
		case 106://j
		{   FrameOprtor->DIYMultiFusion(HDRimages);
		imshow("DIYhdri for align pics", FrameOprtor->HDRresult);
		waitKey(0);
		break;
		}
		case 107://k
		{
			FrameOprtor->WriteCurrentHDR_result();
			break;
		}
		case 108://l
		{
			FrameOprtor->DIY_Seam_Carving_Based_MultiFusion(images);
			break;
		}
		case 109://m
		{
			FrameOprtor->DIYMultiFusion_AdSaturation_HSL_Write(images);
			break;
		}
		case 110://n
		{
			Eresult = FrameOprtor->HDRresult;
			WindowUI_OP();
			break;
		}
		case 111://o
		{
			FrameOprtor->JpegDIYMultiFusion_AdSaturation_HSL(images);
			break;
		}
		case 112://p
		{
			delete FrameOprtor;
			SequenceOperation();
			exit(0);
			break;
		}
		case 113://q
		{  now = false;
		break;
		}
		default:
		{
			cout << " Input key Wrong,try agin." << endl;
			break;
		}
		}cv::destroyAllWindows();
	}
	delete FrameOprtor;
}
	//��JPEGͼ����Ȩ�صĵײ�
void MacroBlockProcess(int n)
{
	if (!this->JPEGWeights.size())
	ReadJPEGWeight();//�����޵д�bug��jpegȨ��ͼbmp��ʽ��λ���8��������ȴ����ͨ��
	if (this->JPEGWeights[n].rows * 8 != Framesize.height || JPEGWeights[n].cols * 8 != Framesize.width)
	{
		cout << endl << "Fatal Error in MacroBlockProcess : JPEG size and Frame size not match!" << endl << " JPEG: " << JPEGWeights[n].size() << " Framesize: " << Framesize.width<< " " << Framesize.height << endl;
		system("pause");
		exit(1);
	}
	if (this->JPEGWeights.size() != this->Framenum)
	{
		cout << endl << "Fatal Error in MacroBlockProcess : JPEG num and Frame num not match!" << endl << " JPEG: " << JPEGWeights.size() << " Framenum: " << Framenum << endl;
		system("pause");
		exit(1);
	}
	resize(this->JPEGWeights[n], this->JPEGWeights[n], cv::Size(), 8, 8);
	//cout << JPEGWeights[n].channels() << endl;
	//���������ͱ任�Ƿ�������ݽض�
	this->JPEGWeights[n].convertTo(JPEGWeights[n], CV_32FC3, 1 / 255.0, 0);
	cv::Mat imageTransform1;
	if (n != 0)
	{
		if (HomographyReadyFlag)
		warpPerspective(JPEGWeights[n], JPEGWeights[n], Homographys[n - 1], cv::Size(JPEGWeights[n].cols, JPEGWeights[n].rows));
		cvtColor(JPEGWeights[n], imageTransform1, CV_RGB2GRAY); 
		this->GOPWeightMat.push_back(imageTransform1);//��˹�������������Ȩ�ر���Ϊfloat��		
	}
	else 
	{
		cvtColor(JPEGWeights[n], imageTransform1, CV_RGB2GRAY);
		this->GOPWeightMat.push_back(imageTransform1);//��˹�������������Ȩ�ر���Ϊfloat��		
	
	}
}
