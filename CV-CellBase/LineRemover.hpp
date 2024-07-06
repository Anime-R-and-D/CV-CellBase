#include "Filter.hpp"

class LineRemover : public Filter {
	std::vector<cv::Vec3b> lineColors;
	std::vector<cv::Vec3b> excludedColors;
	int maxTimes;

	std::vector<cv::Point> collectLinePositions(cv::Mat srcImg) {
		const int width = srcImg.cols;
		const int height = srcImg.rows;

		std::vector<cv::Point> returnVec;
		returnVec.reserve(static_cast<size_t>(width) * height);

		for (int imgX = 0; imgX < width; imgX++) {
			for (int imgY = 0; imgY < height; imgY++) {
				const cv::Vec3b srcColor = srcImg.at<cv::Vec3b>(imgY, imgX);
				for (const auto& lineColor : lineColors) {
					if (srcColor == lineColor) {
						returnVec.emplace_back(imgX, imgY);
						break;
					}
				}
			}
		}

		return returnVec;
	}

	__forceinline bool __replaceColor(const cv::Mat& srcImg, cv::Mat& dstImg, const cv::Point& position, const std::vector<cv::Vec3b>& excludedColors, const int kernelY, const int kernelX) {
		const int sampleY = position.y + kernelY;
		const int sampleX = position.x + kernelX;
		if (sampleY < 0 || srcImg.rows <= sampleY || sampleX < 0 || srcImg.cols <= sampleX) {
			return false;
		}

		const cv::Vec3b srcColor = srcImg.at<cv::Vec3b>(sampleY, sampleX);

		for (const auto& excludedColor : excludedColors) {
			if (srcColor == excludedColor) {
				return false;
			}
		}

		dstImg.at<cv::Vec3b>(position) = srcColor;

		return true;

	}

	__forceinline bool replaceColor(const cv::Mat& srcImg, cv::Mat& dstImg, const cv::Point& position, const std::vector<cv::Vec3b>& excludedColors) {
		for (int kernelY = -1; kernelY <= 1; kernelY++) {
			for (int kernelX = -1; kernelX <= 1; kernelX++) {
				if (kernelY == 0 && kernelX == 0) {
					continue;
				}

				bool isReplaced = __replaceColor(srcImg, dstImg, position, excludedColors, kernelY, kernelX);
				if (isReplaced) {
					return true;
				}
			}
		}

		return false;
	}

	std::pair<cv::Mat, std::vector<cv::Point>> _apply(cv::Mat srcImg, const std::vector<cv::Point>& linePositions) {
		std::vector<cv::Vec3b> excludedColors = this->excludedColors;
		excludedColors.insert(excludedColors.end(), lineColors.begin(), lineColors.end());

		auto dstImg = srcImg.clone();

		std::vector<cv::Point> newLinePositions;
		newLinePositions.reserve(linePositions.size());

		for (const auto& linePosition : linePositions) {
			bool isReplaced = replaceColor(srcImg, dstImg, linePosition, excludedColors);
			if (isReplaced == false) {
				newLinePositions.push_back(linePosition);
			}
		}

		return std::make_pair(dstImg, newLinePositions);
	}

public:
	LineRemover(std::vector<cv::Vec3b> _lineColors, std::vector< cv::Vec3b> _excludedColors, int _maxTimes) : lineColors(_lineColors), excludedColors(_excludedColors), maxTimes(_maxTimes) {}

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
