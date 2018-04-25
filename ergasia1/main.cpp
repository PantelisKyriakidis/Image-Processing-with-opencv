#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

#include <stdlib.h> 
#include <stdio.h> 
#include <iostream>

using namespace std;
using namespace cv;


Mat image; //(global)original image

//my quicksort+swapfunction
void swap(int i, int j, int a[]) {
	int temp = a[i];
	a[i] = a[j];
	a[j] = temp;
}
void quicksort(int arr[], int left, int right) {
	int min = (left + right) / 2;
	//cout << "QS:" << left << "," << right << "\n";
	
	int i = left;
	int j = right;
	int pivot = arr[min];

	while (left<j || i<right)
	{
		while (arr[i]<pivot)
			i++;
		while (arr[j]>pivot)
			j--;

		if (i <= j) {
			swap(i, j, arr);
			i++;
			j--;
		}
		else {
			if (left<j)
				quicksort(arr, left, j);
			if (i<right)
				quicksort(arr, i, right);
			return;
		}
	}
}

void mouseHandler(int event, int x, int y, int flags, void* param) {
	Vec3b p = image.at<Vec3b>(y, x); // BGR pixel color values
	int hsi = (p[0] + p[1] + p[2]) / 3; //average of three colors
	char HSI[15];
	char rgb[45];
	
	sprintf_s(HSI, "HSI = %d", hsi);
	sprintf_s(rgb, "Red = %d    Green = %d    Blue = %d", p[2] , p[1] , p[0]);
	Mat info(100, 410, CV_8UC3, Scalar(255, 255, 255));
	putText(info, HSI, cvPoint(150, 33), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	putText(info, rgb, cvPoint(0, 66), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(150, 20, 80), 1, CV_AA);
	imshow("information", info);
}


void add_salt_pepper(Mat in, int r) {

	for (int i = 0; i < in.rows; i++) {
		for (int k = 0; k <in.cols; k++) {
			int random = rand() % r + 1;
			if (random == 1) {
				in.at<uchar>(i, k) = 255;
			}
			if (random == 2) {
				in.at<uchar>(i, k) = 0;
			}

		}

	}
}

//for double type!
void convolution(Mat denoising, Mat noised, double window[][3]) // I need the noised image beacuse denoising image is changing  
{                                   //from the algorithm and therefore i cant use it for the convolution
	double sum, a;
	for (int i = 0; i<denoising.rows - 2; i++)
	{
		for (int j = 0; j<denoising.cols - 2; j++)
		{
			sum = 0;
			for (int k = 0; k<3; k++)
			{
				for (int l = 0; l<3; l++)
				{
					a = noised.at<double>(i + k, j + l);
					sum += a*window[k][l];
				}
			}
			denoising.at<double>(i, j) = sum;
		}
	}
}

// 2-D convolution for MEAN filtering
void mean(Mat denoising, Mat noised , double window[][3]) // I need the noised image beacuse denoising image is changing  
{                                                        //from the algorithm and therefore i cant use it for the convolution
	double sum, a;
	for (int i = 0; i<denoising.rows - 2; i++)
	{
		for (int j = 0; j<denoising.cols - 2; j++)
		{
			sum = 0;
			for (int k = 0; k<3; k++)
			{
				for (int l = 0; l<3; l++)
				{
					a = noised.at<uchar>(i + k, j + l);
					sum += a*window[k][l];
				}
			}
			denoising.at<uchar>(i, j) = sum;
		}
	}
}

int giveMetheMedian(int arr[], int length)
{
	quicksort(arr, 0, length - 1);
	int median = arr[length / 2];
	return median;
}

//algorithm for median filter / adaptive thresholding
void medianFilter(Mat denoising,bool AdaptiveThres)
{
	int threshold;//for adaptive thresholding
	int arr[9]; // array for the median algorithm

	
	for (int i = 1; i < denoising.rows - 1; i++)
	{
		for (int j = 1; j < denoising.cols - 1; j++)
		{
			arr[0] = denoising.at<uchar>(i - 1, j - 1);
			arr[1] = denoising.at<uchar>(i - 1, j);
			arr[2] = denoising.at<uchar>(i - 1, j + 1);
			arr[3] = denoising.at<uchar>(i, j - 1);
			arr[4] = denoising.at<uchar>(i, j);
			arr[5] = denoising.at<uchar>(i, j + 1);
			arr[6] = denoising.at<uchar>(i + 1, j - 1);
			arr[7] = denoising.at<uchar>(i + 1, j);
			arr[8] = denoising.at<uchar>(i + 1, j + 1);

			if (AdaptiveThres == false) {
				denoising.at<uchar>(i, j) = giveMetheMedian(arr, 9);
			}
			else {
				threshold = giveMetheMedian(arr, 9) ;
				if (denoising.at<uchar>(i, j) > threshold)
				{
					denoising.at<uchar>(i, j) = 255;
				}
				else
				{
					denoising.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	
}

void edgeDetection(Mat src,Mat _final, Mat grey1, Mat grey2, double hor[][3], double ver[][3]) {
	// filters
	convolution(grey1, src, hor);
	convolution(grey2, src, ver);
	double x,y; //pixel values
	// calculating final
	for (int i = 0; i < grey1.rows; i++)
	{
		for (int j = 0; j < grey2.cols; j++)
		{
			x = grey1.at<double>(i, j);
			y = grey2.at<double>(i, j);
			//final result
			_final.at<double>(i, j) = sqrt(x*x + y*y); ;
		}
	}
}



int main(int argc, char** argv)
{

	string fullPath;
	fullPath = "data/peppers.png";

	image = imread(fullPath, CV_LOAD_IMAGE_COLOR);   // Read the file

	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	namedWindow("original", WINDOW_AUTOSIZE); moveWindow("original", -10, -10);// Create a window for display.
	setMouseCallback("original", mouseHandler, 0); //mouse listener
	imshow("original", image);//Show original image      
	
	namedWindow("information");//pixel info
	
	waitKey(0);
	//.................RGB..to..Grayscale...............................................
	Mat gimage;
	cvtColor(image, gimage, CV_BGR2GRAY); // RGB to Grayscale conversion
	
	//show
	namedWindow("greyscale", WINDOW_AUTOSIZE); moveWindow("grayscale", 20, 60);
	imshow("greyscale", gimage);  
	imwrite("data/GrayscalePeppers.png", gimage);//Show and save grayscale image
	
	waitKey(0);
	//.................Adding..noise....................................................
	Mat gimageClone1 = gimage.clone(); //Clone1 for Gaussian noise. (gimage remains for the other exercises)
	Mat gimageClone2 = gimage.clone();//Clone2 for Salt and pepper. (gimage remains for the other exercises)
    //.................GAUSSIAN.NOISE
	Mat noise1 = Mat(gimage.size(), CV_64F);
	randn(noise1, 0, 0.05);        //creating random noise
	normalize(gimageClone1, gimageClone1, 0.0, 1.0, CV_MINMAX, CV_64F);
	gimageClone1 += noise1;
	gimageClone1.convertTo(gimageClone1, CV_8UC3, 255.0);//convert again to uchar
    //show
	namedWindow("greyscale with Gaussian noise", WINDOW_AUTOSIZE); moveWindow("grayscale with Gaussian noise", 20, 60);
	imshow("greyscale with Gaussian noise", gimageClone1);
	imwrite("data/GaussianNoise.png", gimageClone1);//save Gaussian noise image.
	waitKey(0);
	//.................SALT.AND.PEPPER.NOISE
	//(black and white pixels) (i will use a 10% for white an 10% for black
	add_salt_pepper(gimageClone2, 20);//function
	//show
	namedWindow("greyscale with Salt and Pepper", WINDOW_AUTOSIZE); moveWindow("grayscale with Salt and Pepper", 20, 60);
	imshow("greyscale with Salt and Pepper", gimageClone2);
	imwrite("data/SaltandPepper.png", gimageClone2);//save Salt and Pepper noise image.

	//....................................DeNoising(filters).........................................................
	//....................MEAN..FILTER
	Mat meanDenoisedGaussian = gimageClone1.clone();  //Clone for mean filter on gaussian noise 
	Mat meanDenoisedSnP = gimageClone2.clone();       //Clone for mean filter on Salt and Pepper noise
	
	double window[3][3] = { { 1 / 9.0 , 1 / 9.0 , 1 / 9.0 } ,
	{ 1 / 9.0 , 1 / 9.0 , 1 / 9.0 } ,
	{ 1 / 9.0 , 1 / 9.0 , 1 / 9.0 } };

	waitKey(0);
    //......mean filter for Gaussian noise
	mean(meanDenoisedGaussian, gimageClone1 , window);
	//show 
	namedWindow("Mean filter on Gaussian noise", WINDOW_AUTOSIZE); moveWindow("Mean filter on Gaussian noise", 20, 60);
	imshow("Mean filter on Gaussian noise", meanDenoisedGaussian);
	imwrite("data/MeanfilterOnGauss.png", meanDenoisedGaussian);//save meanfilter on Gauss image.
	
	waitKey(0);
	//.......mean filter for Salt and Pepper noise
	mean(meanDenoisedSnP, gimageClone2 , window);
	//show 
	namedWindow("Mean filter on SnP noise", WINDOW_AUTOSIZE); moveWindow("Mean filter on SnP noise", 60, 20);
	imshow("Mean filter on SnP noise", meanDenoisedSnP);
	imwrite("data/MeanfilterOnSnP.png", meanDenoisedSnP);//save meanfilter on Salt n Pepper image.

	//....................MEDIAN..FILTER
	Mat medianDenoisedGaussian = gimageClone1.clone();  //Clone for median filter on gaussian noise 
	Mat medianDenoisedSnP = gimageClone2.clone();       //Clone for median filter on Salt and Pepper noise
	
	waitKey(0);
    //......median filter for Gaussian noise
	medianFilter(medianDenoisedGaussian, false);
	//show 
	namedWindow("Median filter on Gaussian noise", WINDOW_AUTOSIZE); moveWindow("Median filter on Gaussian noise", 20, 60);
	imshow("Median filter on Gaussian noise", medianDenoisedGaussian);
	imwrite("data/MedianfilterOnGauss.png", medianDenoisedGaussian);//save medianfilter on Gauss image.

	waitKey(0);
	//......median filter for SnP noise
	medianFilter(medianDenoisedSnP, false);
	
    //show 
	namedWindow("Median filter on SnP noise", WINDOW_AUTOSIZE); moveWindow("Median filter on SnP noise", 60, 20);
	imshow("Median filter on SnP noise", medianDenoisedSnP);
	imwrite("data/MedianfilterOnSnP.png", medianDenoisedSnP);//save medianfilter on Salt n Pepper image.
	
	
	//....................................
	//release uneccesary Mats
	gimageClone1.release();
	gimageClone2.release();
	noise1.release();
	meanDenoisedGaussian.release();
	meanDenoisedSnP.release();
	medianDenoisedGaussian.release();
	medianDenoisedSnP.release();
	//.....................................IMAGE..SEGMENTATION....................................
	waitKey(0);
	//.....Adaptive..thresholding
	Mat adaptive = gimage.clone();
	medianFilter(adaptive, true);

	//show
	imshow("Adaptive Thresholding", adaptive);
	imwrite("data/AdaptiveThresholding.png", adaptive);

	//....................................EDGE..DETECTION......................................................................
	waitKey(0);
	//............Prewitt
	gimage.convertTo(gimage, CV_64FC1, 1.0 / 255.0);
	Mat grey1 = gimage.clone();
	//grey1.convertTo(grey1, CV_32F, 1.0 / 255.0)
	Mat grey2 = gimage.clone();
	//grey2.convertTo(grey2, CV_32F, 1.0 / 255.0)
    Mat _final=gimage.clone();
	
	double Horizontal[3][3] = { { -1 , 0 , 1 } ,
	{ -1 , 0 , 1 } ,
	{ -1 , 0 , 1 } };

	double Vertical[3][3] = { { 1 , 1 , 1 } ,
	{ 0 , 0 , 0 } ,
	{ -1 , -1 , -1 } };
	edgeDetection(gimage, _final, grey1, grey2, Horizontal, Vertical);
	imshow("Prewitt on grayscale", _final);
	_final.convertTo(_final, CV_8UC3, 255);
	imwrite("data/Prewitt.png", _final);

	waitKey(0);
	//..................sobel
	Mat grey3 = gimage.clone();
	//grey1.convertTo(grey1, CV_32F, 1.0 / 255.0)
	Mat grey4 = gimage.clone();
	//grey2.convertTo(grey2, CV_32F, 1.0 / 255.0)
	Mat _final2 = gimage.clone();

	double Horizontal2[3][3] = { { -1 , 0 , 1 } ,
	{ -2 , 0 , 2 } ,
	{ -1 , 0 , 1 } };

	double Vertical2[3][3] = { { 1 , 2 , 1 } ,
	{ 0 , 0 , 0 } ,
	{ -1 , -2 , -1 } };

	edgeDetection(gimage, _final2, grey3, grey4, Horizontal2, Vertical2);
	imshow("Sobel on grayscale",_final2);
	_final2.convertTo(_final2, CV_8UC3, 255);
	imwrite("data/Sobel.png", _final2);
	

	cout << "DONE!!!" << endl;
	waitKey(0);
	return 0;
}