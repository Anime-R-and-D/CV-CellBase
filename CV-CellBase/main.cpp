#include "Filter.hpp"

Mat characterCellProcessing(Mat srcImg) {
	vector clothesColors = {
		Vec3b(111, 105, 161),
		Vec3b(144, 160, 130),
		Vec3b(163, 168, 165),
		Vec3b(150, 155, 156),
	};

	vector hairColors1 = {
		Vec3b(41, 38, 40),
		Vec3b(97, 57, 70)
	};

	vector hairColors2 = {
		Vec3b(2, 2, 1),
		Vec3b(44, 17, 10)
	};

	vector eyeColors = {
		Vec3b(28, 9, 11),
		Vec3b(112, 72, 87)
	};

	vector targetColorsList = { clothesColors, hairColors1, hairColors2, eyeColors };

	Mat layer_1 = applyFilters(
		srcImg, {
			make_shared<::CellBlur>(20.0f, 21, targetColorsList),
		});

	Mat layer_2 = applyFilters(
		srcImg, {
			make_shared<::CellBlur>(20.0f, 21, targetColorsList),
			make_shared<LineRemover>(Vec3b(4,2,10), Vec3b(255,255,255), 100),
		});

	Mat layer_3 = applyFilters(
		srcImg, {
			make_shared<LineOnly>(),
		});

	Mat layer_1_2 = applyLayersWithAlpha(layer_1, layer_2, 0.7);
	Mat layer_1_2_3 = applyLayersWithAlpha(layer_1_2, layer_3, 0.3);

	return layer_1_2_3;
}

void characterCellProcessingMovie(const string& srcImgsPathPattern, const string& dstMoviePath) {
	vector<cv::String> srcImgPaths;
	vector<Mat> dstImgs;
	cv::glob(srcImgsPathPattern, srcImgPaths, true);

	for (const string& srcImgPath : srcImgPaths) {
		Mat srcImg = imread(srcImgPath);
		if (srcImg.empty()) {
			throw "No image found! " + srcImgPath;
		}

		Mat dstImg = characterCellProcessing(srcImg);
		dstImgs.push_back(dstImg);
		imshow("CharacterCellProcessingMovie", dstImg);
		waitKey(1);
	}

	int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	cv::VideoWriter writer(dstMoviePath, fourcc, 12, dstImgs[0].size(), true);
	if (!writer.isOpened()) {
		throw "writer not opened: " + dstMoviePath;
	}

	for (const Mat& dstImg : dstImgs) {
		writer.write(dstImg);
	}

	writer.release();
}

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
}

int main()
{
	string imagePath = "src.png";
	Mat srcImage = imread(imagePath);
	if (srcImage.data == NULL)
	{
		throw "No image found! " + imagePath;
	}
	cv::imshow("Source", srcImage);
	Mat line = applyFilters(
		srcImage, {
			make_shared<LineOnly>(),
			make_shared<AveragingBlur>(2,2),
			
		});

	Mat newImg = applyChoke(line, 10);
	//newImg = applyChoke(newImg, 3);
	
	cv::imshow("LineOnly", line);
	cv::imshow("Choke", newImg);
	// imshow("AveragingBlur", AveragingBlur(3, 3).apply(srcImage));
	// imshow("GaussianBlur", ::GaussianBlur(2.0f, 5).apply(srcImage));
	// imshow("SobelX", SobelX().apply(srcImage));
	// imshow("SobelY", SobelY().apply(srcImage));
	// imshow("SobelAbsXY", SobelAbsXY().apply(srcImage));

	// imshow("CharacterCellProcessing", characterCellProcessing(srcImage));
	//characterCellProcessingMovie("movie_test/*.png", "results.avi");

	// auto chalkImage = chalkFilter(srcImage);
	// imshow("ChalkFilter", chalkImage);
	// auto charImagePtr = static_cast<void*>(&chalkImage);
	// setMouseCallback("ChalkFilter", onMouse, (charImagePtr));

	cv::waitKey(0);
	return 0;
}
