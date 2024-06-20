#include "Filter.hpp"

int main() {
	string imagePath = "src.png";
	Mat srcImage = imread(imagePath);
	if (srcImage.data == NULL) {
		throw "No image found! " + imagePath;
	}
	imshow("Source", srcImage);

	Mat dstImage = applyFilters(
		srcImage, {
			make_shared<SobelAbsXY>(),
			make_shared<::GaussianBlur>(2.0f,5),
		});
	imshow("Result", dstImage);

	//imshow("AveragingBlur", AveragingBlur(3, 3).apply(srcImage));
	//imshow("GaussianBlur", ::GaussianBlur(2.0f, 5).apply(srcImage));
	//imshow("SobelX", SobelX().apply(srcImage));
	//imshow("SobelY", SobelY().apply(srcImage));
	//imshow("SobelAbsXY", SobelAbsXY().apply(srcImage));

	waitKey(0);
	return 0;
}
