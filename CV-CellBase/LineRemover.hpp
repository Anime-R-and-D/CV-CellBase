#include "Filter.hpp"

class LineRemover : public Filter {
	cv::Vec3b lineColor;
	cv::Vec3b backgroundColor;
	int maxTimes;

	std::vector<cv::Vec2i> collectLinePositions(cv::Mat srcImg) {
		std::vector<cv::Vec2i> returnVec;

		const int width = srcImg.cols;
		const int height = srcImg.rows;

		for (int imgX = 0; imgX < width; imgX++) {
			for (int imgY = 0; imgY < height; imgY++) {
				const cv::Vec3b srcColor = srcImg.at<cv::Vec3b>(imgY, imgX);
				if (srcColor == lineColor) {
					returnVec.push_back(cv::Vec2i(imgY, imgX));
				}
			}
		}

		return returnVec;
	}

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

				const cv::Vec3b srcColor = srcImg.at<cv::Vec3b>(sampleY, sampleX);
				if (srcColor != lineColor && srcColor != backgroundColor) {
					return srcColor;
				}
			}
		}

		return lineColor;
	}

	std::pair<cv::Mat, int> _apply(cv::Mat srcImg, std::vector<cv::Vec2i> linePositions) {
		auto dstImg = srcImg.clone();

		int notFoundCount = 0;

		for (const auto& linePosition : linePositions) {
			const int y = linePosition[0];
			const int x = linePosition[1];

			auto dstColor = getRepaintColor(srcImg, x, y);

			if (dstColor != lineColor) {
				dstImg.at<cv::Vec3b>(y, x) = dstColor;
			}
			else {
				notFoundCount++;
			}
		}

		return std::make_pair(dstImg, notFoundCount);
	}

public:
	LineRemover(cv::Vec3b _lineColor, cv::Vec3b _backgroundColor, int _maxTimes) : lineColor(_lineColor), backgroundColor(backgroundColor), maxTimes(_maxTimes) {}

	cv::Mat apply(cv::Mat srcImg) {
		cv::Mat img = srcImg;
		auto linePositions = collectLinePositions(srcImg);

		for (int i = 0; i < maxTimes; i++) {
			auto ret = _apply(img, linePositions);
			img = ret.first;
			if (ret.second == 0) { break; }
		}

		return img;
	}
};
