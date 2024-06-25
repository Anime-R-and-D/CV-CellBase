#pragma once
#include <cmath>
#include <numbers>
#include <span>
#include <filesystem>

#include <opencv2/opencv.hpp>

#ifdef _DEBUG
#pragma comment (lib, "opencv_world4100d.lib")
#else
#pragma comment (lib, "opencv_world4100.lib")
#endif

using namespace cv;
using namespace std;

template<typename T, typename int N>
inline Vec<T, N> abs(Vec<T, N> src) {
	Vec<T, N> dst;

	for (int i = 0; i < N; i++) {
		dst[i] = abs(src[i]);
	}

	return dst;
}

template<typename T, typename int N>
inline Vec<T, N> max(Vec<T, N> src1, Vec<T, N> src2) {
	Vec<T, N> dst;

	for (int i = 0; i < N; i++) {
		dst[i] = max(src1[i], src2[i]);
	}

	return dst;
}

class Filter {
public:
	virtual Mat apply(Mat srcImg) = 0;

};

class LinearFilter :public Filter {
public:
	Mat_<float> kernel;

protected:
	LinearFilter(int kernel_width, int kernel_height) {
		kernel = Mat_<float>(kernel_height, kernel_width);
	}

public:
	Mat apply(Mat srcImg) {
		auto dstImg = Mat(srcImg.size(), srcImg.type());

		const int kernelCenterY = kernel.rows / 2;
		const int kernelCenterX = kernel.cols / 2;

		for (int imgY = 0; imgY < srcImg.rows; imgY++) {
			for (int imgX = 0; imgX < srcImg.cols; imgX++) {
				Vec3f dstImgPixel(0, 0, 0);

				for (int kernelY = 0; kernelY < kernel.rows; kernelY++) {
					for (int kernelX = 0; kernelX < kernel.cols; kernelX++) {
						auto imgSampleY = clamp(imgY + kernelY - kernelCenterY, 0, srcImg.rows - 1);
						auto imgSampleX = clamp(imgX + kernelX - kernelCenterX, 0, srcImg.cols - 1);
						auto srcImgPixel = srcImg.at<Vec3b>(imgSampleY, imgSampleX);

						auto weight = kernel(kernelY, kernelX);
						dstImgPixel += static_cast<Vec3f>(srcImgPixel) * weight;
					}
				}

				dstImg.at<Vec3b>(imgY, imgX) = dstImgPixel;
			}
		}

		return dstImg;
	}
};

class AveragingBlur : public LinearFilter {
public:
	AveragingBlur(int kernel_width, int kernel_height) : LinearFilter(kernel_width, kernel_height) {
		float weight = 1.0f / (kernel_width * kernel_height);

		for (int kernelY = 0; kernelY < kernel.rows; kernelY++) {
			for (int kernelX = 0; kernelX < kernel.cols; kernelX++) {
				kernel(kernelY, kernelX) = weight;
			}
		}
	}
};

class SobelX : public LinearFilter {
public:
	SobelX() : LinearFilter(3, 3) {
		float weights[3][3] = { {1, 0, -1},
								{2, 0, -2},
								{1, 0, -1} };

		for (int kernelY = 0; kernelY < kernel.rows; kernelY++) {
			for (int kernelX = 0; kernelX < kernel.cols; kernelX++) {
				kernel(kernelY, kernelX) = weights[kernelY][kernelX];
			}
		}
	}
};

class SobelY : public LinearFilter {
public:
	SobelY() : LinearFilter(3, 3) {
		float weights[3][3] = { {1, 2, 1},
								{0, 0, 0},
								{-1, -2, -1} };

		for (int kernelY = 0; kernelY < kernel.rows; kernelY++) {
			for (int kernelX = 0; kernelX < kernel.cols; kernelX++) {
				kernel(kernelY, kernelX) = weights[kernelY][kernelX];
			}
		}
	}
};

class SobelAbsXY :public Filter {
private:
	Mat_<float> sobelX = SobelX().kernel;
	Mat_<float> sobelY = SobelY().kernel;

public:
	Mat apply(Mat srcImg) {
		auto dstImg = Mat(srcImg.size(), srcImg.type());

		const int kernelCenterY = sobelX.rows / 2;
		const int kernelCenterX = sobelX.cols / 2;

		for (int imgY = 1; imgY < srcImg.rows - 1; imgY++) {
			for (int imgX = 1; imgX < srcImg.cols - 1; imgX++) {
				Vec3f dstImgPixelX(0, 0, 0), dstImgPixelY(0, 0, 0);
				for (int kernelY = 0; kernelY < sobelX.rows; kernelY++) {
					for (int kernelX = 0; kernelX < sobelX.cols; kernelX++) {
						auto imgSampleY = imgY + kernelY - kernelCenterY;
						auto imgSampleX = imgX + kernelX - kernelCenterX;
						auto srcImgPixel = srcImg.at<Vec3b>(imgSampleY, imgSampleX);

						auto weightX = sobelX(kernelY, kernelX);
						auto weightY = sobelY(kernelY, kernelX);

						dstImgPixelX += static_cast<Vec3f>(srcImgPixel) * weightX;
						dstImgPixelY += static_cast<Vec3f>(srcImgPixel) * weightY;
					}
				}
				dstImg.at<Vec3b>(imgY, imgX) = max(abs(dstImgPixelX), abs(dstImgPixelY));
			}
		}

		return dstImg;
	}
};

class GaussianBlur : public LinearFilter {
public:
	GaussianBlur(float sigma, int size) : LinearFilter(size, size) {
		constexpr float pi = static_cast<float>(numbers::pi);

		float gauss_total = 0.0f;
		int center = size / 2;

		for (int kernelY = 0; kernelY < size; kernelY++) {
			for (int kernelX = 0; kernelX < size; kernelX++) {
				int lengthY = center - kernelY;
				int lengthX = center - kernelX;

				float part1 = 1 / (2 * pi * sigma * sigma);
				float part2 = exp(-(lengthX * lengthX + lengthY * lengthY) / (2 * sigma * sigma));
				float weight = part1 * part2;

				kernel(kernelY, kernelX) = weight;
				gauss_total += weight;
			}
		}

		for (int kernelY = 0; kernelY < size; kernelY++) {
			for (int kernelX = 0; kernelX < size; kernelX++) {
				kernel(kernelY, kernelX) /= gauss_total;
			}
		}
	}
};

class CellBlur : public Filter {
public:
	Mat_<float> kernel;
	vector<vector<Vec3b>> targets;

	CellBlur(float sigma, int size, vector<vector<Vec3b>> _targets) : kernel(::GaussianBlur(sigma, size).kernel), targets(_targets) {	}

	Mat_<bool> _createTargetFlagImg(Mat srcImg, vector<Vec3b> target) {
		auto dstImg = Mat_<bool>(srcImg.size());

		for (int imgY = 0; imgY < srcImg.rows; imgY++) {
			for (int imgX = 0; imgX < srcImg.cols; imgX++) {
				dstImg(imgY, imgX) = find(target.begin(), target.end(), srcImg.at<Vec3b>(imgY, imgX)) != target.end();
			}
		}

		return dstImg;
	}

	Mat _apply(Mat _srcImg, const Mat_<bool>& targetFlagImg) {
		Mat_<Vec3f> srcImg = _srcImg;
		auto dstImg = Mat(_srcImg.size(), _srcImg.type());

		const int kernelCenterY = kernel.rows / 2;
		const int kernelCenterX = kernel.cols / 2;
		for (int imgY = 0; imgY < srcImg.rows; imgY++) {
			for (int imgX = 0; imgX < srcImg.cols; imgX++) {
				if (targetFlagImg(imgY, imgX)) {
					Vec3f dstImgPixel(0, 0, 0);
					float weightSum = 0.0f;

					for (int kernelY = 0; kernelY < kernel.rows; kernelY++) {
						for (int kernelX = 0; kernelX < kernel.cols; kernelX++) {
							auto imgSampleY = clamp(imgY + kernelY - kernelCenterY, 0, srcImg.rows - 1);
							auto imgSampleX = clamp(imgX + kernelX - kernelCenterX, 0, srcImg.cols - 1);
							auto weight = kernel.at<float>(kernelY, kernelX);

							if (targetFlagImg(imgSampleY, imgSampleX)) {
								auto srcImgPixel = srcImg(imgSampleY, imgSampleX);
								dstImgPixel += srcImgPixel * weight;
								weightSum += weight;
							}
						}
					}
					dstImg.at<Vec3b>(imgY, imgX) = dstImgPixel / weightSum;
				}
				else {
					dstImg.at<Vec3b>(imgY, imgX) = _srcImg.at<Vec3b>(imgY, imgX);
				}
			}
		}
		return dstImg;
	}

	Mat apply(Mat srcImg) {
		auto img = srcImg;

		for (auto target : targets) {
			auto targetFlagImg = _createTargetFlagImg(img, target);
			img = _apply(img, targetFlagImg);
		}

		return img;
	}
};

class LineOnly : public Filter {

	Mat apply(Mat srcImg) {
		auto dstImg = Mat(srcImg.size(), srcImg.type());
		for (int imgY = 1; imgY < srcImg.rows - 1; imgY++) {
			for (int imgX = 1; imgX < srcImg.cols - 1; imgX++) {
				if (srcImg.at<Vec3b>(imgY, imgX) == Vec3b(4, 2, 10)) {
					dstImg.at<Vec3b>(imgY, imgX) = Vec3b(4, 2, 10);
				}
				else {
					dstImg.at<Vec3b>(imgY, imgX) = Vec3b(255, 255, 255);
				}
			}
		}
		return dstImg;
	}

};

class LineRemover : public Filter {
	Vec3b lineColor;
	Vec3b backgroundColor;
	int maxTimes;

	auto getRepaintColor(Mat srcImg, int x, int y) {
		const int width = srcImg.cols;
		const int height = srcImg.rows;

		for (int kernelY = -1; kernelY <= 1; kernelY++) {
			for (int kernelX = -1; kernelX <= 1; kernelX++) {
				const int sampleY = y + kernelY;
				const int sampleX = x + kernelX;
				if (sampleY < 0 || height <= sampleY || sampleX < 0 || width <= sampleX) {
					continue;
				}

				auto srcColor = srcImg.at<Vec3b>(sampleY, sampleX);
				if (srcColor != lineColor && srcColor != backgroundColor) {
					return srcColor;
				}
			}
		}

		return lineColor;
	}

	pair<Mat, int> _apply(Mat srcImg) {
		auto dstImg = srcImg.clone();

		int width = srcImg.cols;
		int height = srcImg.rows;

		int notFoundCount = 0;

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				auto srcColor = srcImg.at<Vec3b>(y, x);
				if (srcColor == lineColor) {
					auto dstColor = getRepaintColor(srcImg, x, y);
					if (dstColor != lineColor) { dstImg.at<Vec3b>(y, x) = dstColor; }
					else { notFoundCount++; }
				}
			}
		}

		return make_pair(dstImg, notFoundCount);
	}

public:
	LineRemover(Vec3b _lineColor, Vec3b _backgroundColor, int _maxTimes) : lineColor(_lineColor), backgroundColor(backgroundColor), maxTimes(_maxTimes) {}

	Mat apply(Mat srcImg) {
		auto img = srcImg;

		for (int i = 0; i < maxTimes; i++) {
			auto ret = _apply(img);
			if (ret.second == 0) { break; }
			else { img = ret.first; }
		}

		return img;
	}
};

class Choke : public Filter {
public:
	double chokeMatte1;

	Choke(double _chokeMatte1) : chokeMatte1(_chokeMatte1) {}

private:
	Mat applyChokeY(Mat img, double chokeMatte) {
		auto dstImg = Mat(img.size(), img.type());

		for (int imgX = 0; imgX < img.cols - 1; imgX++) {
			for (int imgY = 0; imgY < img.rows - 1; imgY++) {

				int altered = 0;
				if (img.at<Vec3b>(imgY, imgX) != Vec3b(255, 255, 255)) {
					if (imgY != 0 && img.at<Vec3b>(imgY - 1, imgX) == Vec3b(255, 255, 255)) {
						altered++;
						for (int k = 0; k < chokeMatte; k++) {
							if (imgY + k > img.rows - 1) {
								dstImg.at<Vec3b>(img.rows - 1, imgX) = Vec3b(255, 255, 255);
							}
							else {
								dstImg.at<Vec3b>(imgY + k, imgX) = Vec3b(255, 255, 255);
							}
						}
						imgY += chokeMatte;
					}
					else if (img.at<Vec3b>(imgY + 1, imgX) == Vec3b(255, 255, 255)) {
						altered++;
						for (int k = 0; k < chokeMatte; k++) {
							if (imgY - k < 0) {
								dstImg.at<Vec3b>(0, imgX) = Vec3b(255, 255, 255);
							}
							else {
								dstImg.at<Vec3b>(imgY - k, imgX) = Vec3b(255, 255, 255);
							}
						}
					}

					if (altered == 0) {
						dstImg.at<Vec3b>(imgY, imgX) = img.at<Vec3b>(imgY, imgX);
					}
				}

				else {
					dstImg.at<Vec3b>(imgY, imgX) = img.at<Vec3b>(imgY, imgX);
				}
			}
		}

		return dstImg;
	}

	Mat applyChokeX(Mat img, double chokeMatte) {
		auto dstImg = Mat(img.size(), img.type());

		for (int imgY = 0; imgY < img.rows - 1; imgY++) {
			for (int imgX = 0; imgX < img.cols - 1; imgX++) {
				int altered = 0;
				if (img.at<Vec3b>(imgY, imgX) != Vec3b(255, 255, 255)) {
					if (img.at<Vec3b>(imgY, imgX - 1) == Vec3b(255, 255, 255)) {
						altered++;
						for (int k = 0; k < chokeMatte; k++) {
							if (imgX + k > img.cols - 1) {
								dstImg.at<Vec3b>(imgY, img.cols - 1) = Vec3b(255, 255, 255);
							}
							else {
								dstImg.at<Vec3b>(imgY, imgX + k) = Vec3b(255, 255, 255);
							}
						}
						imgX += chokeMatte;
					}
					else if (img.at<Vec3b>(imgY, imgX + 1) == Vec3b(255, 255, 255)) {
						altered++;
						for (int k = 0; k < chokeMatte; k++) {
							if (imgX - k < img.cols) {
								dstImg.at<Vec3b>(imgY, img.cols) = Vec3b(255, 255, 255);
							}
							else {
								dstImg.at<Vec3b>(imgY, imgX - k) = Vec3b(255, 255, 255);
							}
						}

					}

					if (altered == 0) {
						dstImg.at<Vec3b>(imgY, imgX) = img.at<Vec3b>(imgY, imgX);
					}
				}
				else {
					dstImg.at<Vec3b>(imgY, imgX) = img.at<Vec3b>(imgY, imgX);
				}
			}
		}

		return dstImg;
	}

public:
	Mat apply(Mat img) {
		int chokeMatte = chokeMatte1 / 2;
		auto chokedXImg = applyChokeX(img, chokeMatte);
		return applyChokeY(chokedXImg, chokeMatte);
	}
};

Mat applyLayers(vector<Mat> srcImgs) {
	auto dstImg = Mat(srcImgs.at(0).size(), srcImgs.at(0).type());
	for (int imgY = 0; imgY < srcImgs.at(0).rows; imgY++) {
		for (int imgX = 0; imgX < srcImgs.at(0).cols; imgX++) {
			int whiteBuff = 0;
			int totalB = 0;
			int totalG = 0;
			int totalR = 0;
			for (Mat img : srcImgs) {
				if (img.at<Vec3b>(imgY, imgX) == Vec3b(255, 255, 255)) {
					whiteBuff++;
				}
				else {
					totalB += img.at<Vec3b>(imgY, imgX)[0];
					totalG += img.at<Vec3b>(imgY, imgX)[1];
					totalR += img.at<Vec3b>(imgY, imgX)[2];
				}
			}
			if (whiteBuff == srcImgs.size()) {
				dstImg.at<Vec3b>(imgY, imgX) = Vec3b(255, 255, 255);
			}
			else {
				dstImg.at<Vec3b>(imgY, imgX) = Vec3b(totalB / (srcImgs.size() - whiteBuff), totalG / (srcImgs.size() - whiteBuff), totalR / (srcImgs.size() - whiteBuff));
			}
		}

	}
	return dstImg;
}

Mat applyLayersWithAlpha(Mat bg, Mat fg, double alpha) {
	if (alpha > 1) {
		alpha == 1;
	}
	auto dstImg = Mat(bg.size(), bg.type());
	for (int imgY = 0; imgY < bg.rows; imgY++) {
		for (int imgX = 0; imgX < bg.cols; imgX++) {
			if (fg.at<Vec3b>(imgY, imgX) == Vec3b(255, 255, 255)) {
				dstImg.at<Vec3b>(imgY, imgX) = bg.at<Vec3b>(imgY, imgX);

			}
			else if (bg.at<Vec3b>(imgY, imgX) == Vec3b(255, 255, 255)) {
				dstImg.at<Vec3b>(imgY, imgX) = fg.at<Vec3b>(imgY, imgX);
			}
			else {
				int bgB = bg.at<Vec3b>(imgY, imgX)[0] * (1 - alpha);
				int bgG = bg.at<Vec3b>(imgY, imgX)[1] * (1 - alpha);
				int bgR = bg.at<Vec3b>(imgY, imgX)[2] * (1 - alpha);

				int fgB = fg.at<Vec3b>(imgY, imgX)[0] * alpha;
				int fgG = fg.at<Vec3b>(imgY, imgX)[1] * alpha;
				int fgR = fg.at<Vec3b>(imgY, imgX)[2] * alpha;
				dstImg.at<Vec3b>(imgY, imgX) = Vec3b(bgB + fgB, bgG + fgG, bgR + fgR);
			}
		}
	}
	return dstImg;
}

Mat applyAlpha(Mat image, double alpha) {
	if (alpha > 1) {
		alpha == 1;
	}
	auto dstImg = Mat(image.size(), image.type());
	for (int imgY = 0; imgY < image.rows - 1; imgY++) {
		for (int imgX = 0; imgX < image.cols - 1; imgX++) {
			if (image.at<Vec4b>(imgY, imgX) == Vec4b(255, 255, 255, 255)) {
				dstImg.at<Vec4b>(imgY, imgX) = Vec4b(255, 255, 255, 255);

			}
			else {
				int newB = image.at<Vec4b>(imgY, imgX)[0];
				int newG = image.at<Vec4b>(imgY, imgX)[1];
				int newR = image.at<Vec4b>(imgY, imgX)[2];
				int newA = image.at<Vec4b>(imgY, imgX)[3];

				dstImg.at<Vec4b>(imgY, imgX) = Vec4b(newB * alpha + (255 * (1 - alpha)), newG * alpha + (255 * (1 - alpha)), newR * alpha + (255 * (1 - alpha)), newA * alpha + (255 * (1 - alpha)));
			}
		}

	}

	return dstImg;
}

Mat applyFilters(Mat srcImg, const span<const shared_ptr<Filter>> filters) {
	auto img = srcImg;

	for (auto filter : filters) {
		img = filter->apply(img);
	}

	return img;
}

Mat applyFilters(Mat srcImg, const initializer_list<shared_ptr<Filter>> filters) {
	span _span(filters.begin(), filters.size());
	return applyFilters(srcImg, _span);
}
