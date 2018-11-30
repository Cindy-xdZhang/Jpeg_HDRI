#ifndef _HDR_H
#define _HDR_H
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/stitching.hpp>
#include<vector>
#include<iostream>
#include <io.h>  
#include "MultiFusion.h"
#include "stdafx.h"
#include<conio.h>
#include <math.h>
#include<direct.h>
using namespace cv;
using namespace std;
template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}
template<typename T> string toString(const T& t)
{
	ostringstream oss;  //创建一个格式化输出流
	oss << t;             //把值传递如流中
	return oss.str();
}
//搜索路劲由IMAGE_SEARCH_PATH制定。//如IMAGE_SEARCH_PATH="inputimage/8/*.bmp"字符串里注意不要出现空格！！！
//ReadSequenceImage:在某文件夹内读取图片的程序，不需要在意图片名字、数量等，
//但char * IMAGE_SEARCH_PATH 后缀名如果多于3个字母则需要修改函数
void ReadSequenceImage( vector<Mat>&images);
#endif
//MAT.TYPE=1:8UC1 5:32FC1