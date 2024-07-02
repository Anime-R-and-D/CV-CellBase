#include "Filter.hpp"
#include "CellBlur.hpp"

#ifdef _DEBUG
#pragma comment (lib, "opencv_world4100d.lib")
#else
#pragma comment (lib, "opencv_world4100.lib")
#endif

cv::Mat characterCellProcessing(cv::Mat srcImg) {
	std::vector clothesColors = {
		cv::Vec3b(111, 105, 161),
		cv::Vec3b(144, 160, 130),
		cv::Vec3b(163, 168, 165),
		cv::Vec3b(150, 155, 156),
	};

	std::vector hairColors1 = {
		cv::Vec3b(41, 38, 40),
		cv::Vec3b(97, 57, 70)
	};

	std::vector hairColors2 = {
		cv::Vec3b(2, 2, 1),
		cv::Vec3b(44, 17, 10)
	};

	std::vector eyeColors = {
		cv::Vec3b(28, 9, 11),
		cv::Vec3b(112, 72, 87)
	};

	std::vector targetColorsList = { clothesColors, hairColors1, hairColors2, eyeColors };

	cv::Mat layer_1 = applyFilters(
		srcImg, {
			std::make_shared<::CellBlur>(20.0f, 21, targetColorsList),
		});

	cv::Mat layer_2 = applyFilters(
		srcImg, {
			std::make_shared<::CellBlur>(20.0f, 21, targetColorsList),
			std::make_shared<LineRemover>(cv::Vec3b(4,2,10), cv::Vec3b(255,255,255), 100),
		});

	cv::Mat layer_3 = applyFilters(
		srcImg, {
			std::make_shared<LineOnly>(),
		});

	cv::Mat layer_1_2 = applyLayersWithAlpha(layer_1, layer_2, 0.7);
	cv::Mat layer_1_2_3 = applyLayersWithAlpha(layer_1_2, layer_3, 0.3);

	return layer_1_2_3;
}

void characterCellProcessingMovie(const std::string& srcImgsPathPattern, const std::string& dstMoviePath) {
	std::vector<cv::String> srcImgPaths;
	std::vector<cv::Mat> dstImgs;
	cv::glob(srcImgsPathPattern, srcImgPaths, true);

	for (const std::string& srcImgPath : srcImgPaths) {
		cv::Mat srcImg = cv::imread(srcImgPath);
		if (srcImg.empty()) {
			throw "No image found! " + srcImgPath;
		}

		cv::Mat dstImg = characterCellProcessing(srcImg);
		dstImgs.push_back(dstImg);
		imshow("CharacterCellProcessingMovie", dstImg);
		cv::waitKey(1);
	}

	int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	cv::VideoWriter writer(dstMoviePath, fourcc, 12, dstImgs[0].size(), true);
	if (!writer.isOpened()) {
		throw "writer not opened: " + dstMoviePath;
	}

	for (const cv::Mat& dstImg : dstImgs) {
		writer.write(dstImg);
	}

	writer.release();
}

cv::Mat chalkFilter(cv::Mat srcImage) {
	auto colorLine = applyFilters(
		srcImage, {
			std::make_shared<::SobelAbsXY>(),
			std::make_shared<::GaussianBlur>(2.0f, 3),
		});

	cv::Mat grayLine;
	cvtColor(colorLine, grayLine, cv::COLOR_BGR2GRAY);
	cv::Mat mask = grayLine != 0;

	cv::Mat noise(colorLine.size(), colorLine.type());
	cv::randn(noise, 300, 200);

	cv::Mat_<cv::Vec3i> iNoisedColorLine = static_cast<cv::Mat_<cv::Vec3i>>(colorLine) + static_cast<cv::Mat_<cv::Vec3i>>(noise);
	cv::Mat_<cv::Vec3b> bNoisedColorLine = iNoisedColorLine & cv::Vec3i(255, 255, 255);

	cv::Mat noisedGrayLine;
	cvtColor(bNoisedColorLine, noisedGrayLine, cv::COLOR_BGR2GRAY);

	return noisedGrayLine & mask;
}

void chokedLine(cv::Mat srcImage) {
	auto line = applyFilters(
		srcImage, {
			std::make_shared<LineOnly>(),
			std::make_shared<AveragingBlur>(2,2),
		});
	cv::imshow("LineOnly", line);

	auto chokedLine = applyFilters(
		line, {
			std::make_shared<Choke>(10),
		});
	cv::imshow("Choke", chokedLine);
}

static void onMouse(int event, int x, int y, int f, void* param) {
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		auto image = static_cast<cv::Mat*>(param);
		auto pixel = image->at<uchar>(y, x);
		std::cout << static_cast<int>(pixel) << std::endl;
	}
}

int main()
{
	std::string imagePath = "src.png";
	cv::Mat srcImage = cv::imread(imagePath);
	if (srcImage.data == NULL)
	{
		throw "No image found! " + imagePath;
	}
	cv::imshow("Source", srcImage);

	// cv::imshow("AveragingBlur", AveragingBlur(3, 3).apply(srcImage));
	// cv::imshow("GaussianBlur", ::GaussianBlur(2.0f, 5).apply(srcImage));
	// cv::imshow("SobelX", SobelX().apply(srcImage));
	// cv::imshow("SobelY", SobelY().apply(srcImage));
	// cv::imshow("SobelAbsXY", SobelAbsXY().apply(srcImage));

	// cv::imshow("CharacterCellProcessing", characterCellProcessing(srcImage));
	// characterCellProcessingMovie("movie_test/*.png", "results.avi");

	// auto chalkImage = chalkFilter(srcImage);
	// cv::imshow("ChalkFilter", chalkImage);
	// auto charImagePtr = static_cast<void*>(&chalkImage);
	// cv::setMouseCallback("ChalkFilter", onMouse, (charImagePtr));

	chokedLine(srcImage);

	cv::waitKey(0);
	return 0;
}
