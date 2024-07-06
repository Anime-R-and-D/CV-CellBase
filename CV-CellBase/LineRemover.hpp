#include "Filter.hpp"

class LineRemover : public Filter {
	cv::Vec3b lineColor;
	cv::Vec3b backgroundColor;
	int maxTimes;

	std::vector<cv::Point> collectLinePositions(cv::Mat srcImg) {
		const int width = srcImg.cols;
		const int height = srcImg.rows;

		std::vector<cv::Point> returnVec;
		returnVec.reserve(static_cast<size_t>(width) * height);

		for (int imgX = 0; imgX < width; imgX++) {
			for (int imgY = 0; imgY < height; imgY++) {
				const cv::Vec3b srcColor = srcImg.at<cv::Vec3b>(imgY, imgX);
				if (srcColor == lineColor) {
					returnVec.emplace_back(imgX, imgY);
				}
			}
		}

		return returnVec;
	}

	auto getRepaintColor(cv::Mat srcImg, const cv::Point& position) {
		const int width = srcImg.cols;
		const int height = srcImg.rows;

		for (int kernelY = -1; kernelY <= 1; kernelY++) {
			for (int kernelX = -1; kernelX <= 1; kernelX++) {
				const int sampleY = position.y + kernelY;
				const int sampleX = position.x + kernelX;
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

	std::pair<cv::Mat, std::vector<cv::Point>> _apply(cv::Mat srcImg, const std::vector<cv::Point>& linePositions) {
		auto dstImg = srcImg.clone();

		std::vector<cv::Point> newLinePositions;
		newLinePositions.reserve(linePositions.size());

		for (const auto& linePosition : linePositions) {
			auto dstColor = getRepaintColor(srcImg, linePosition);

			if (dstColor != lineColor) {
				dstImg.at<cv::Vec3b>(linePosition) = dstColor;
			}
			else {
				newLinePositions.push_back(linePosition);
			}
		}

		return std::make_pair(dstImg, newLinePositions);
	}

public:
	LineRemover(cv::Vec3b _lineColor, cv::Vec3b _backgroundColor, int _maxTimes) : lineColor(_lineColor), backgroundColor(backgroundColor), maxTimes(_maxTimes) {}

	cv::Mat apply(cv::Mat srcImg) {
		cv::Mat img = srcImg;
		auto linePositions = collectLinePositions(srcImg);

		for (int i = 0; i < maxTimes && 0 < linePositions.size(); i++) {
			auto ret = _apply(img, linePositions);
			img = ret.first;
			linePositions = ret.second;
		}

		return img;
	}
};
