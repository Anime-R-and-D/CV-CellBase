#include "Filter.hpp"

int main()
{
	string imagePath = "src.png";
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
		Vec3b(97, 57, 70)};

	vector hairColors2 = {
		Vec3b(2, 2, 1),
		Vec3b(44, 17, 10)};

	vector eyeColors = {
		Vec3b(28, 9, 11),
		Vec3b(112, 72, 87)};

	vector targetColorsList = {clothesColors, hairColors1, hairColors2, eyeColors};

	Mat dstImage = applyFilters(
		srcImage, {
					  make_shared<::CellBlur>(20.0f, 20, targetColorsList),
					  make_shared<LineRemover>(Vec3b(4,2,10), Vec3b(255,255,255), 100),
					  make_shared<::GaussianBlur>(10.0f, 5),
				  });

	cv::imshow("Result", dstImage);

	// imshow("AveragingBlur", AveragingBlur(3, 3).apply(srcImage));
	// imshow("GaussianBlur", ::GaussianBlur(2.0f, 5).apply(srcImage));
	// imshow("SobelX", SobelX().apply(srcImage));
	// imshow("SobelY", SobelY().apply(srcImage));
	// imshow("SobelAbsXY", SobelAbsXY().apply(srcImage));

	cv::waitKey(0);
	return 0;
}
