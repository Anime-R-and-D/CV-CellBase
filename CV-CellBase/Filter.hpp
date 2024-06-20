#pragma once
#include <cmath>
#include <numbers>
#include <span>

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
