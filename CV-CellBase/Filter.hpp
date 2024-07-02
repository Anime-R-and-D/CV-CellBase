#pragma once
#include <cmath>
#include <numbers>
#include <span>
#include <filesystem>

#include <opencv2/opencv.hpp>

template<typename T, typename int N>
inline cv::Vec<T, N> abs(cv::Vec<T, N> src) {
	cv::Vec<T, N> dst;

	for (int i = 0; i < N; i++) {
		dst[i] = abs(src[i]);
	}

	return dst;
}

template<typename T, typename int N>
inline cv::Vec<T, N> max(cv::Vec<T, N> src1, cv::Vec<T, N> src2) {
	cv::Vec<T, N> dst;

	for (int i = 0; i < N; i++) {
		dst[i] = std::max(src1[i], src2[i]);
	}

	return dst;
}

class Filter {
public:
	virtual cv::Mat apply(cv::Mat srcImg) = 0;

};

class LinearFilter :public Filter {
public:
	cv::Mat_<float> kernel;

protected:
	LinearFilter(int kernel_width, int kernel_height) {
		kernel = cv::Mat_<float>(kernel_height, kernel_width);
	}

public:
	cv::Mat apply(cv::Mat srcImg) {
		auto dstImg = cv::Mat(srcImg.size(), srcImg.type());

		const int kernelCenterY = kernel.rows / 2;
		const int kernelCenterX = kernel.cols / 2;

		for (int imgY = 0; imgY < srcImg.rows; imgY++) {
			for (int imgX = 0; imgX < srcImg.cols; imgX++) {
				cv::Vec3f dstImgPixel(0, 0, 0);

				for (int kernelY = 0; kernelY < kernel.rows; kernelY++) {
					for (int kernelX = 0; kernelX < kernel.cols; kernelX++) {
						auto imgSampleY = std::clamp(imgY + kernelY - kernelCenterY, 0, srcImg.rows - 1);
						auto imgSampleX = std::clamp(imgX + kernelX - kernelCenterX, 0, srcImg.cols - 1);
						auto srcImgPixel = srcImg.at<cv::Vec3b>(imgSampleY, imgSampleX);

						auto weight = kernel(kernelY, kernelX);
						dstImgPixel += static_cast<cv::Vec3f>(srcImgPixel) * weight;
					}
				}

				dstImg.at<cv::Vec3b>(imgY, imgX) = dstImgPixel;
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
	cv::Mat_<float> sobelX = SobelX().kernel;
	cv::Mat_<float> sobelY = SobelY().kernel;

public:
	cv::Mat apply(cv::Mat srcImg) {
		auto dstImg = cv::Mat(srcImg.size(), srcImg.type());

		const int kernelCenterY = sobelX.rows / 2;
		const int kernelCenterX = sobelX.cols / 2;

		for (int imgY = 1; imgY < srcImg.rows - 1; imgY++) {
			for (int imgX = 1; imgX < srcImg.cols - 1; imgX++) {
				cv::Vec3f dstImgPixelX(0, 0, 0), dstImgPixelY(0, 0, 0);
				for (int kernelY = 0; kernelY < sobelX.rows; kernelY++) {
					for (int kernelX = 0; kernelX < sobelX.cols; kernelX++) {
						auto imgSampleY = imgY + kernelY - kernelCenterY;
						auto imgSampleX = imgX + kernelX - kernelCenterX;
						auto srcImgPixel = srcImg.at<cv::Vec3b>(imgSampleY, imgSampleX);

						auto weightX = sobelX(kernelY, kernelX);
						auto weightY = sobelY(kernelY, kernelX);

						dstImgPixelX += static_cast<cv::Vec3f>(srcImgPixel) * weightX;
						dstImgPixelY += static_cast<cv::Vec3f>(srcImgPixel) * weightY;
					}
				}
				dstImg.at<cv::Vec3b>(imgY, imgX) = max(abs(dstImgPixelX), abs(dstImgPixelY));
			}
		}

		return dstImg;
	}
};

class GaussianBlur : public LinearFilter {
public:
	GaussianBlur(float sigma, int size) : LinearFilter(size, size) {
		constexpr float pi = static_cast<float>(std::numbers::pi);

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

class LineOnly : public Filter {
	cv::Mat apply(cv::Mat srcImg) {
		auto dstImg = cv::Mat(srcImg.size(), srcImg.type());
		for (int imgY = 1; imgY < srcImg.rows - 1; imgY++) {
			for (int imgX = 1; imgX < srcImg.cols - 1; imgX++) {
				if (srcImg.at<cv::Vec3b>(imgY, imgX) == cv::Vec3b(4, 2, 10)) {
					dstImg.at<cv::Vec3b>(imgY, imgX) = cv::Vec3b(4, 2, 10);
				}
				else {
					dstImg.at<cv::Vec3b>(imgY, imgX) = cv::Vec3b(255, 255, 255);
				}
			}
		}
		return dstImg;
	}

};

class LineRemover : public Filter {
	cv::Vec3b lineColor;
	cv::Vec3b backgroundColor;
	int maxTimes;

	auto getRepaintColor(cv::Mat srcImg, int x, int y) {
		const int width = srcImg.cols;
		const int height = srcImg.rows;

		for (int kernelY = -1; kernelY <= 1; kernelY++) {
			for (int kernelX = -1; kernelX <= 1; kernelX++) {
				const int sampleY = y + kernelY;
				const int sampleX = x + kernelX;
				if (sampleY < 0 || height <= sampleY || sampleX < 0 || width <= sampleX) {
					continue;
				}

				auto srcColor = srcImg.at<cv::Vec3b>(sampleY, sampleX);
				if (srcColor != lineColor && srcColor != backgroundColor) {
					return srcColor;
				}
			}
		}

		return lineColor;
	}

	std::pair<cv::Mat, int> _apply(cv::Mat srcImg) {
		auto dstImg = srcImg.clone();

		int width = srcImg.cols;
		int height = srcImg.rows;

		int notFoundCount = 0;

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				auto srcColor = srcImg.at<cv::Vec3b>(y, x);
				if (srcColor == lineColor) {
					auto dstColor = getRepaintColor(srcImg, x, y);
					if (dstColor != lineColor) { dstImg.at<cv::Vec3b>(y, x) = dstColor; }
					else { notFoundCount++; }
				}
			}
		}

		return std::make_pair(dstImg, notFoundCount);
	}

public:
	LineRemover(cv::Vec3b _lineColor, cv::Vec3b _backgroundColor, int _maxTimes) : lineColor(_lineColor), backgroundColor(backgroundColor), maxTimes(_maxTimes) {}

	cv::Mat apply(cv::Mat srcImg) {
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
	int chokeMatte1;

	Choke(int _chokeMatte1) : chokeMatte1(_chokeMatte1) {}

private:
	cv::Mat applyChokeY(cv::Mat img, int chokeMatte) {
		auto dstImg = cv::Mat(img.size(), img.type());

		for (int imgX = 0; imgX < img.cols - 1; imgX++) {
			for (int imgY = 0; imgY < img.rows - 1; imgY++) {

				int altered = 0;
				if (img.at<cv::Vec3b>(imgY, imgX) != cv::Vec3b(255, 255, 255)) {
					if (imgY != 0 && img.at<cv::Vec3b>(imgY - 1, imgX) == cv::Vec3b(255, 255, 255)) {
						altered++;
						for (int k = 0; k < chokeMatte; k++) {
							if (imgY + k > img.rows - 1) {
								dstImg.at<cv::Vec3b>(img.rows - 1, imgX) = cv::Vec3b(255, 255, 255);
							}
							else {
								dstImg.at<cv::Vec3b>(imgY + k, imgX) = cv::Vec3b(255, 255, 255);
							}
						}
						imgY += chokeMatte;
					}
					else if (img.at<cv::Vec3b>(imgY + 1, imgX) == cv::Vec3b(255, 255, 255)) {
						altered++;
						for (int k = 0; k < chokeMatte; k++) {
							if (imgY - k < 0) {
								dstImg.at<cv::Vec3b>(0, imgX) = cv::Vec3b(255, 255, 255);
							}
							else {
								dstImg.at<cv::Vec3b>(imgY - k, imgX) = cv::Vec3b(255, 255, 255);
							}
						}
					}

					if (altered == 0) {
						dstImg.at<cv::Vec3b>(imgY, imgX) = img.at<cv::Vec3b>(imgY, imgX);
					}
				}

				else {
					dstImg.at<cv::Vec3b>(imgY, imgX) = img.at<cv::Vec3b>(imgY, imgX);
				}
			}
		}

		return dstImg;
	}

	cv::Mat applyChokeX(cv::Mat img, int chokeMatte) {
		auto dstImg = cv::Mat(img.size(), img.type());

		for (int imgY = 0; imgY < img.rows - 1; imgY++) {
			for (int imgX = 0; imgX < img.cols - 1; imgX++) {
				int altered = 0;
				if (img.at<cv::Vec3b>(imgY, imgX) != cv::Vec3b(255, 255, 255)) {
					if (img.at<cv::Vec3b>(imgY, imgX - 1) == cv::Vec3b(255, 255, 255)) {
						altered++;
						for (int k = 0; k < chokeMatte; k++) {
							if (imgX + k > img.cols - 1) {
								dstImg.at<cv::Vec3b>(imgY, img.cols - 1) = cv::Vec3b(255, 255, 255);
							}
							else {
								dstImg.at<cv::Vec3b>(imgY, imgX + k) = cv::Vec3b(255, 255, 255);
							}
						}
						imgX += chokeMatte;
					}
					else if (img.at<cv::Vec3b>(imgY, imgX + 1) == cv::Vec3b(255, 255, 255)) {
						altered++;
						for (int k = 0; k < chokeMatte; k++) {
							if (imgX - k < img.cols) {
								dstImg.at<cv::Vec3b>(imgY, img.cols) = cv::Vec3b(255, 255, 255);
							}
							else {
								dstImg.at<cv::Vec3b>(imgY, imgX - k) = cv::Vec3b(255, 255, 255);
							}
						}

					}

					if (altered == 0) {
						dstImg.at<cv::Vec3b>(imgY, imgX) = img.at<cv::Vec3b>(imgY, imgX);
					}
				}
				else {
					dstImg.at<cv::Vec3b>(imgY, imgX) = img.at<cv::Vec3b>(imgY, imgX);
				}
			}
		}

		return dstImg;
	}

public:
	cv::Mat apply(cv::Mat img) {
		int chokeMatte = chokeMatte1 / 2;
		auto chokedXImg = applyChokeX(img, chokeMatte);
		return applyChokeY(chokedXImg, chokeMatte);
	}
};

cv::Mat applyLayers(std::vector<cv::Mat> srcImgs) {
	auto dstImg = cv::Mat(srcImgs.at(0).size(), srcImgs.at(0).type());
	int srcCount = static_cast<int>(srcImgs.size());

	for (int imgY = 0; imgY < srcImgs.at(0).rows; imgY++) {
		for (int imgX = 0; imgX < srcImgs.at(0).cols; imgX++) {
			int whiteBuff = 0;
			int totalB = 0;
			int totalG = 0;
			int totalR = 0;
			for (cv::Mat img : srcImgs) {
				if (img.at<cv::Vec3b>(imgY, imgX) == cv::Vec3b(255, 255, 255)) {
					whiteBuff++;
				}
				else {
					totalB += img.at<cv::Vec3b>(imgY, imgX)[0];
					totalG += img.at<cv::Vec3b>(imgY, imgX)[1];
					totalR += img.at<cv::Vec3b>(imgY, imgX)[2];
				}
			}
			if (whiteBuff == srcImgs.size()) {
				dstImg.at<cv::Vec3b>(imgY, imgX) = cv::Vec3b(255, 255, 255);
			}
			else {
				dstImg.at<cv::Vec3b>(imgY, imgX) = cv::Vec3b(totalB / (srcCount - whiteBuff), totalG / (srcCount - whiteBuff), totalR / (srcCount - whiteBuff));
			}
		}

	}
	return dstImg;
}

cv::Mat applyLayersWithAlpha(cv::Mat bg, cv::Mat fg, double alpha) {
	if (alpha > 1) {
		alpha = 1;
	}

	auto dstImg = cv::Mat(bg.size(), bg.type());
	for (int imgY = 0; imgY < bg.rows; imgY++) {
		for (int imgX = 0; imgX < bg.cols; imgX++) {
			if (fg.at<cv::Vec3b>(imgY, imgX) == cv::Vec3b(255, 255, 255)) {
				dstImg.at<cv::Vec3b>(imgY, imgX) = bg.at<cv::Vec3b>(imgY, imgX);

			}
			else if (bg.at<cv::Vec3b>(imgY, imgX) == cv::Vec3b(255, 255, 255)) {
				dstImg.at<cv::Vec3b>(imgY, imgX) = fg.at<cv::Vec3b>(imgY, imgX);
			}
			else {
				auto bgPixel = bg.at<cv::Vec3b>(imgY, imgX);
				auto fgPixel = fg.at<cv::Vec3b>(imgY, imgX);
				dstImg.at<cv::Vec3b>(imgY, imgX) = bgPixel * (1 - alpha) + fgPixel * alpha;
			}
		}
	}
	return dstImg;
}

cv::Mat applyAlpha(cv::Mat image, double alpha) {
	if (alpha > 1) {
		alpha = 1;
	}

	auto dstImg = cv::Mat(image.size(), image.type());
	for (int imgY = 0; imgY < image.rows - 1; imgY++) {
		for (int imgX = 0; imgX < image.cols - 1; imgX++) {
			if (image.at<cv::Vec4b>(imgY, imgX) == cv::Vec4b(255, 255, 255, 255)) {
				dstImg.at<cv::Vec4b>(imgY, imgX) = cv::Vec4b(255, 255, 255, 255);

			}
			else {
				dstImg.at<cv::Vec4b>(imgY, imgX) = image.at<cv::Vec4b>(imgY, imgX) * alpha + cv::Vec4b(255, 255, 255, 255) * (1 - alpha);
			}
		}

	}

	return dstImg;
}

cv::Mat applyFilters(cv::Mat srcImg, const std::span<const std::shared_ptr<Filter>> filters) {
	auto img = srcImg;

	for (auto filter : filters) {
		img = filter->apply(img);
	}

	return img;
}

cv::Mat applyFilters(cv::Mat srcImg, const std::initializer_list<std::shared_ptr<Filter>> filters) {
	std::span _span(filters.begin(), filters.size());
	return applyFilters(srcImg, _span);
}
