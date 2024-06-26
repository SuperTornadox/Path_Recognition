
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;
using namespace std;


int main()
{
	Mat M = imread("C:\\Users\\Xuchen He\\Desktop\\test2.jpg");

	//先将我们感兴趣的部分图像转换为鸟瞰图 这样可以更方便在较低的视角看清较远的路况

	Point2f srcVertices[4];
	Point2f dstVertices[4];

	//兴趣区域边界点
	srcVertices[0] = Point(700, 605);
	srcVertices[1] = Point(890, 605);
	srcVertices[2] = Point(1760, 1030);
	srcVertices[3] = Point(20, 1030);

	//目标边界点
	dstVertices[0] = Point(0, 0);
	dstVertices[1] = Point(640, 0);
	dstVertices[2] = Point(640, 480);
	dstVertices[3] = Point(0, 480);

    //进行坐标系变换
	Mat perspectiveMatrix = getPerspectiveTransform(srcVertices, dstVertices);
	Mat dst(480, 640, CV_8U); 
	Mat invertedPerspectiveMatrix;
	invert(perspectiveMatrix, invertedPerspectiveMatrix);
	warpPerspective(M, dst, perspectiveMatrix, dst.size(), INTER_LINEAR, BORDER_CONSTANT);

	imshow("step1", dst);

	Mat GR(480,640,CV_8U);
	//因此在这里建立一个尺寸相同的灰度图像
	cvtColor(dst,GR,COLOR_RGB2GRAY);
	//这里将图像转变为灰色

	imshow("step2", GR);

	//接下来做一些简单的处理
	//part1 高斯模糊

	Mat GR2(480, 640, CV_8U);
	GaussianBlur(GR,GR2,Size(11,11),0,0);//注意这里高斯模糊的模板长宽必须是奇数

	imshow("step3", GR2);
	//part2 开闭操作
	Mat GR3(480, 640, CV_8U);
	dilate(GR2,GR3,(10,10));
	Mat GR4(480, 640, CV_8U);
	erode(GR3,GR4,(10,10));
	Mat GR5(480, 640, CV_8U);
	morphologyEx(GR4, GR5, MORPH_CLOSE,(10,10));
	imshow("step4", GR3);
	imshow("step5", GR4);
	imshow("step6", GR5);


	Mat BW(480, 640, CV_8U);
	//通过阈值化把车道完全变成黑白的图像
	const int thresholdVal = 150;
	threshold(GR5, BW, thresholdVal, 255, THRESH_BINARY);

	imshow("step7", BW);

	Mat CN(480, 640, CV_8U);
	Canny(BW, CN, 1, 1, 3, false);
	imshow("step8", CN);

	vector<Vec2f> lines;
	HoughLines(CN, lines, 1, CV_PI / 180, 65, 0, 0);//根据我们测试的经验，这里的阈值设为65比较好

	Mat final(1076,1918, CV_8U);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pts1, pts2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pts1.x = cvRound(x0 + 1000 * (-b));
		pts1.y = cvRound(y0 + 1000 * (a));
		pts2.x = cvRound(x0 - 1000 * (-b));
		pts2.y = cvRound(y0 - 1000 * (a));
		line(CN, pts1, pts2, Scalar(255,255,0), 3, LINE_AA);
		
	}


	//官方文档内的画线函数



	imshow("result", CN);
	cout << "success" << endl;
	waitKey(60000);//使得窗口停留一会不要过早消失
	//我们想了很多办法将画的线变回原来的视角，但是这样就没法和车内原先的视野结合，这点还没有完善

}



//shouda sobel

/*
void conv2D(cv::Mat& src, cv::Mat& dst, cv::Mat kernel, int ddepth, cv::Point anchor = cv::Point(-1, -1), int delta = 0, int borderType = cv::BORDER_DEFAULT) {
 cv::Mat  kernelFlip;
 cv::flip(kernel, kernelFlip, -1);
 cv::filter2D(src, dst, ddepth, kernelFlip, anchor, delta, borderType);
}

int main()
{
 Mat srcImage=imread("D:\\×ÀÃæ\\234.jpg");
 imshow("Ô­Í¼", srcImage);
 Mat kernel = (Mat_<char>(3,3)<<1,2,1,0,0,0,-1,-2,-1);
 Mat kerne2 = (Mat_<char>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
 Mat sobelXImage, dstImageX,dstImageY,sobelYImage;
 filter2D(srcImage, sobelXImage, srcImage.depth(), kernel);
 filter2D(srcImage, sobelYImage, srcImage.depth(), kerne2);
 convertScaleAbs(sobelXImage, dstImageX);
 convertScaleAbs(sobelYImage, dstImageY);
 imshow("ÔËËãºóXÍ¼Æ¬", dstImageX);
 imshow("ÔËËãºóYÍ¼Æ¬", dstImageY);
 Mat dstImage;
 addWeighted(dstImageX, 0.5, dstImageY, 0.5, 0, dstImage);
 imshow("sobelËã×Ó×îÖÕÔËËã½á¹û", dstImage);
 waitKey(0);
}
*/

//Sobel hanshu
/*
int main()
{
	Mat Mm=imread("D:\\×ÀÃæ\\234.jpg");
	Mat M;
	vector<Mat> channels;
	split(Mm,channels);
	M=channels.at(1);
	Mat Mx,My,MMM;
	Sobel(M,Mx,-1,1,0);
	Sobel(M,My,-1,0,1);
	addWeighted(Mx,0.5,My,0.5,0,MMM);
	imshow("MMM",MMM);
	waitKey(60000);
}
*/


//Canny

/*
int main()
{
	Mat Mm=imread("D:\\×ÀÃæ\\234.jpg");
	Mat M;
	vector<Mat> channels;
	split(Mm,channels);
	M=channels.at(1);
	Mat MMM;
	Canny(M,MMM,90,60,3,false);

	imshow("MMM",MMM);
	waitKey(60000);
}
*/


/*
int main()
{
	Mat M(800,800,CV_8UC3,Scalar(0,0,0));
	Mat M2(800,800,CV_8UC3,Scalar(0,0,0));
	Mat M3(800,800,CV_8UC3,Scalar(0,0,0));

	line(M,Point(200,700),Point(400,100),Scalar(255,255,255),3,8,0);
	line(M,Point(600,700),Point(400,100),Scalar(255,255,255),3,8,0);
	line(M,Point(80,300),Point(600,700),Scalar(255,255,255),3,8,0);
	line(M,Point(720,300),Point(200,700),Scalar(255,255,255),3,8,0);
	line(M,Point(80,300),Point(720,300),Scalar(255,255,255),3,8,0);
	rectangle(M2,Point(100,100),Point(700,700),Scalar(0,255,0),3,8,0);
	rectangle(M2,Point(200,200),Point(600,600),Scalar(255,0,0),3,8,0);
	rectangle(M2,Point(300,300),Point(500,500),Scalar(0,0,255),3,8,0);
	circle(M3,Point(400,400),100,Scalar(0,255,0),3,8,0);
	circle(M3,Point(400,400),200,Scalar(255,0,0),3,8,0);
	circle(M3,Point(400,400),300,Scalar(0,0,255),3,8,0);
	imshow("Star",M);
	imshow("Rect",M2);
	imshow("circle",M3);
	waitKey(60000);
}
*/
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
