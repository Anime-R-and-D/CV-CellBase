#include "Filter.hpp"

Mat chalkFilter(Mat srcImage) {
	auto colorLine = applyFilters(
		srcImage, {
			make_shared<::SobelAbsXY>(),
			make_shared<::GaussianBlur>(2.0f, 3),
		});

	Mat grayLine;
	cvtColor(colorLine, grayLine, COLOR_BGR2GRAY);
	Mat mask = grayLine != 0;

	cv::Mat noise(colorLine.size(), colorLine.type());
	cv::randn(noise, 300, 200);

	Mat_<Vec3i> iNoisedColorLine = static_cast<Mat_<Vec3i>>(colorLine) + static_cast<Mat_<Vec3i>>(noise);
	Mat_<Vec3b> bNoisedColorLine = iNoisedColorLine & Vec3i(255, 255, 255);

	Mat noisedGrayLine;
	cvtColor(bNoisedColorLine, noisedGrayLine, COLOR_BGR2GRAY);

	return noisedGrayLine & mask;
}

static void onMouse(int event, int x, int y, int f, void* param) {
	if (event == EVENT_LBUTTONDOWN)
	{
		auto image = static_cast<Mat*>(param);
		auto pixel = image->at<uchar>(y, x);
		cout << static_cast<int>(pixel) << endl;
	}
};

int main()
{
	string imagePath = "src.png";
	string moviePath = "movie_test/*.png";


	vector<cv::String> fn;
	vector<cv::Mat> data;
	cv::glob(moviePath, fn, true);

	for (size_t k = 0; k < fn.size(); k++)
	{

		cv::Mat im = cv::imread(fn[k]);
		if (im.empty()) continue;

		data.push_back(im);
	}


	Mat srcImage = imread(imagePath);
	if (srcImage.data == NULL)
	{
		throw "No image found! " + imagePath;
	}
	cv::imshow("Source", srcImage);

	vector clothesColors = {
		Vec3b(111, 105, 161),
		Vec3b(144, 160, 130),
		Vec3b(163, 168, 165),
		Vec3b(150, 155, 156),
	};

	vector hairColors1 = {
		Vec3b(41, 38, 40),
		Vec3b(97, 57, 70) };

	vector hairColors2 = {
		Vec3b(2, 2, 1),
		Vec3b(44, 17, 10) };

	vector eyeColors = {
		Vec3b(28, 9, 11),
		Vec3b(112, 72, 87) };

	vector targetColorsList = { clothesColors, hairColors1, hairColors2, eyeColors };

	vector<cv::Mat> newData;
	for (Mat img : data) {

		Mat layer1 = applyFilters(
			img, {

				//
			   // make_shared<::SobelAbsXY>(),
		  make_shared<::CellBlur>(20.0f, 21, targetColorsList),
		  //  make_shared<::LineOnly>(),
	  /*make_shared<::CellBlur>(20.0f, 21, targetColorsList),
	  make_shared<::GaussianBlur>(10.0f, 11),*/
			});

		Mat layer2 = applyFilters(
			img, {

				//
			   // make_shared<::SobelAbsXY>(),
				make_shared<::CellBlur>(20.0f, 21, targetColorsList),
				make_shared<LineRemover>(Vec3b(4,2,10), Vec3b(255,255,255), 100),
				/*make_shared<::CellBlur>(20.0f, 21, targetColorsList),
				make_shared<::GaussianBlur>(10.0f, 11),*/
			});
		Mat layer3 = applyFilters(
			img, {

			make_shared<LineOnly>(),
			});
		Mat result1 = applyLayersWithAlpha(layer1, layer2, 0.7);
		Mat finalResult = applyLayersWithAlpha(result1, layer3, 0.3);
		newData.push_back(finalResult);

	}

	/*for (Mat img : newData) {
		imshow("newData", img);
		waitKey(0);
	}*/

	int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

	
	cv::VideoWriter writer("results.avi", fourcc, 12, cv::Size(newData.at(0).cols, newData.at(0).rows), true);
	
	if (!writer.isOpened()) {
		cout << "writer not opened" << endl;
	}

	//std::cout << "ttest" << endl;
		//imshow("new results", newData);
	for (int i = 0; i < newData.size(); i++) {
		writer.write(newData.at(i));

		


		

}

	writer.release();

	//Mat layer1 = applyFilters(
	//	srcImage, {
	//				  
	//				  //
	//				 // make_shared<::SobelAbsXY>(),
	//			make_shared<::CellBlur>(20.0f, 21, targetColorsList),
	//				//  make_shared<::LineOnly>(),
	//			/*make_shared<::CellBlur>(20.0f, 21, targetColorsList),
	//			make_shared<::GaussianBlur>(10.0f, 11),*/
	//			  });

	//Mat layer2 = applyFilters(
	//	srcImage, {

	//		//
	//	   // make_shared<::SobelAbsXY>(),
	//		make_shared<::CellBlur>(20.0f, 21, targetColorsList),
	//		make_shared<LineRemover>(Vec3b(4,2,10), Vec3b(255,255,255), 100),
	//		/*make_shared<::CellBlur>(20.0f, 21, targetColorsList),
	//		make_shared<::GaussianBlur>(10.0f, 11),*/
	//	});
	//Mat layer3 = applyFilters(
	//	srcImage, {

	//	make_shared<LineOnly>(),
	//	});

	//cv::imshow("Image2", dstImage1);
	//Mat dstImage2 = applyLayersWithAlpha(dstImage1, dstImage, 0.1);

	// imshow("AveragingBlur", AveragingBlur(3, 3).apply(srcImage));
	// imshow("GaussianBlur", ::GaussianBlur(2.0f, 5).apply(srcImage));
	// imshow("SobelX", SobelX().apply(srcImage));
	// imshow("SobelY", SobelY().apply(srcImage));
	// imshow("SobelAbsXY", SobelAbsXY().apply(srcImage));

	// auto chalkImage = chalkFilter(srcImage);
	// imshow("ChalkFilter", chalkImage);
	// auto charImagePtr = static_cast<void*>(&chalkImage);
	// setMouseCallback("ChalkFilter", onMouse, (charImagePtr));

	cv::waitKey(0);
	return 0;
}
