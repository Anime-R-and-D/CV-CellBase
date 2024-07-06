#include "Filter.hpp"

template <typename T>
class LineRemover : public Filter {
	std::vector<T> lineColors;
	std::vector<T> excludedColors;
	int maxTimes;

	std::vector<cv::Point> collectLinePositions(cv::Mat_<T> srcImg) {
		const int pixelCount = srcImg.cols * srcImg.rows;

		std::vector<cv::Point> linePositions;
		linePositions.reserve(pixelCount);

		auto pixels = reinterpret_cast<T*>(srcImg.data);
		int i = 0;
		for (int imgY = 0; imgY < srcImg.rows; imgY++) {
			for (int imgX = 0; imgX < srcImg.cols; imgX++) {
				const T srcColor = pixels[i++];
				for (const auto& lineColor : lineColors) {
					if (srcColor[0] == lineColor[0] && srcColor[1] == lineColor[1] && srcColor[2] == lineColor[2]) {
						linePositions.emplace_back(imgX, imgY);
						break;
					}
				}
			}
		}

		return linePositions;
	}

	__forceinline bool __replaceColor(const cv::Mat_<T>& srcImg, cv::Mat_<T>& dstImg, const cv::Point& position, const std::vector<T>& excludedColors, const int kernelY, const int kernelX) {
		const int sampleY = position.y + kernelY;
		const int sampleX = position.x + kernelX;
		if (sampleY < 0 || srcImg.rows <= sampleY || sampleX < 0 || srcImg.cols <= sampleX) {
			return false;
		}

		const T srcColor = srcImg(sampleY, sampleX);

		for (const auto& excludedColor : excludedColors) {
			if (srcColor[0] == excludedColor[0] && srcColor[1] == excludedColor[1] && srcColor[2] == excludedColor[2]) {
				return false;
			}
		}

		dstImg(position) = srcColor;

		return true;

	}

	__forceinline bool replaceColor(const cv::Mat_<T>& srcImg, cv::Mat_<T>& dstImg, const cv::Point& position, const std::vector<T>& excludedColors) {
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

	std::vector<cv::Point> _apply(const cv::Mat_<T>& srcImg, cv::Mat_<T>& dstImg, const std::vector<cv::Point>& linePositions) {
		std::vector<T> excludedColors = this->excludedColors;
		excludedColors.insert(excludedColors.end(), lineColors.begin(), lineColors.end());

		std::vector<cv::Point> newLinePositions;
		newLinePositions.reserve(linePositions.size());

		for (const auto& linePosition : linePositions) {
			bool isReplaced = replaceColor(srcImg, dstImg, linePosition, excludedColors);
			if (isReplaced == false) {
				newLinePositions.push_back(linePosition);
			}
		}

		return newLinePositions;
	}

public:
	LineRemover(std::vector<T> _lineColors, std::vector< T> _excludedColors, int _maxTimes) : lineColors(_lineColors), excludedColors(_excludedColors), maxTimes(_maxTimes) {}

	cv::Mat apply(cv::Mat _srcImg) {
		cv::Mat_<T> srcImg = _srcImg;
		cv::Mat_<T> dstImg = srcImg.clone();
		auto linePositions = collectLinePositions(srcImg);

		for (int i = 0; i < maxTimes; i++) {
			auto newLinePositions = _apply(srcImg, dstImg, linePositions);
			if (newLinePositions.size() == 0) {
				break;
			}

			for (const auto& oldLinePosition : linePositions) {
				srcImg(oldLinePosition) = dstImg(oldLinePosition);
			}

			linePositions = std::move(newLinePositions);
		}

		return dstImg;
	}
};

using LineRemover3b = LineRemover<cv::Vec3b>;
using LineRemover3w = LineRemover<cv::Vec3w>;
using LineRemover3f = LineRemover<cv::Vec3f>;
using LineRemover4b = LineRemover<cv::Vec4b>;
using LineRemover4w = LineRemover<cv::Vec4w>;
using LineRemover4f = LineRemover<cv::Vec4f>;

