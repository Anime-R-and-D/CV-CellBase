#include "Filter.hpp"


static void onMouse(int event, int x, int y, int f, void* param) {
	Mat& image = *((Mat*)param);
	Vec4b pixel = image.at<Vec4b>(y, x);
	if (event == EVENT_LBUTTONDOWN)
	{

		cout << pixel << endl;
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

	for (Mat img : newData) {
		imshow("newData", img);
		waitKey(0);
	}

	//int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
	//cv::VideoWriter writer("result.mp4", fourcc, 22.0, cv::Size(newData.at(0).cols, newData.at(0).rows), true);
	//std::cout << "ttest" << endl;
		//imshow("new results", newData);
	/*for (int i = 0; i < newData.size(); i++) {

		writer.write(newData.at(i));

		*/

//}

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
	//Mat chalk1 = applyFilters(
	//	srcImage, {
	//	make_shared<::SobelAbsXY>(),
	//	make_shared<::GaussianBlur>(2.0f, 3),
	//	});
	//cv::Mat noise(chalk1.size(), chalk1.type());
	//cv::randn(noise, 300, 200);
	//
	//
	//Mat chalkResult = applyNoise(chalk1, noise);
	//cv::cvtColor(chalkResult, chalkResult, cv::COLOR_BGR2GRAY);
	//cv::Mat letter = noise - chalk1;
	//cv::cvtColor(letter, letter, cv::COLOR_BGR2GRAY);
	

	
	//cv::imshow("Image2", dstImage1);
	//Mat dstImage2 = applyLayersWithAlpha(dstImage1, dstImage, 0.1);

	//cv::imshow("ChalkTest", chalkResult);
	// imshow("AveragingBlur", AveragingBlur(3, 3).apply(srcImage));
	// imshow("GaussianBlur", ::GaussianBlur(2.0f, 5).apply(srcImage));
	// imshow("SobelX", SobelX().apply(srcImage));
	// imshow("SobelY", SobelY().apply(srcImage));
	// imshow("SobelAbsXY", SobelAbsXY().apply(srcImage));
	//setMouseCallback("ChalkTest", onMouse, &chalk1);
	cv::waitKey(0);
	return 0;
}
