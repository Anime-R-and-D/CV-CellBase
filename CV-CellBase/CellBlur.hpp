#include "Filter.hpp"

constexpr std::int_fast32_t _256 = 256;

class CellBlur : public Filter {
public:
	vector<float> kernel;
	vector<vector<Vec3b>> targets;

	CellBlur(float sigma, int size, vector<vector<Vec3b>> _targets) : kernel(), targets(_targets) {
		float gauss_total = 0.0f;
		int center = size / 2;

		for (int i = 0; i < size; i++) {
			int length = center - i;
			float weight = exp(-static_cast<float>(length * length) / (2 * sigma * sigma));

			kernel.push_back(weight);
			gauss_total += weight;
		}

		for (int i = 0; i < size; i++) {
			kernel[i] /= gauss_total;
		}
	}

	Mat_<int_fast32_t> convertToIntImg(const Mat& srcImg, int* _startNotWhiteX, int* _startNotWhiteY, int* _endNotWhiteX, int* _endNotWhiteY) {
		Mat_<int_fast32_t> iSrcImg(srcImg.size());

		const int size = srcImg.rows * srcImg.cols;
		constexpr int_fast32_t white = 255 + 255 * _256 + 255 * _256 * _256;

		auto srcData = reinterpret_cast<Vec3b*>(srcImg.data);
		auto iSrcData = reinterpret_cast<int_fast32_t*>(iSrcImg.data);

		int startNotWhiteX = srcImg.cols - 1;
		int startNotWhiteY = srcImg.rows - 1;
		int endNotWhiteX = 0;
		int endNotWhiteY = 0;

		int i = 0;
		for (int imgY = 0; imgY < srcImg.rows; imgY++) {
			for (int imgX = 0; imgX < srcImg.cols; imgX++) {
				auto iSrcPixel = srcData[i][0] + srcData[i][1] * _256 + srcData[i][2] * _256 * _256;
				iSrcData[i] = iSrcPixel;
				i++;
				if (iSrcPixel != white) {
					startNotWhiteX = min(startNotWhiteX, imgX);
					startNotWhiteY = min(startNotWhiteY, imgY);
					endNotWhiteX = max(endNotWhiteX, imgX);
					endNotWhiteY = max(endNotWhiteY, imgY);
				}
			}
		}

		*_startNotWhiteX = startNotWhiteX;
		*_startNotWhiteY = startNotWhiteY;
		*_endNotWhiteX = endNotWhiteX;
		*_endNotWhiteY = endNotWhiteY;

		return iSrcImg;
	}

	Mat_<bool> _createTargetFlagImg(Mat_<int_fast32_t> srcImg, vector<Vec3b> _target, int* _startImgX, int* _startImgY, int* _endImgX, int* _endImgY, int startNotWhiteX, int startNotWhiteY, int endNotWhiteX, int endNotWhiteY) {
		auto dstImg = Mat_<bool>(srcImg.size());

		bool inWhite = std::find(_target.begin(), _target.end(), Vec3b(255, 255, 255)) != _target.end();
		if (inWhite == true) {
			startNotWhiteX = 0;
			startNotWhiteY = 0;
			endNotWhiteX = srcImg.cols - 1;
			endNotWhiteY = srcImg.rows - 1;
		}

		const auto srcData = reinterpret_cast<int_fast32_t*>(srcImg.data);
		auto dstData = reinterpret_cast<bool*>(dstImg.data);
		int size = srcImg.rows * srcImg.cols;

		vector<std::int_fast32_t> target;
		for (int i = 0; i < _target.size(); i++) {
			target.push_back(_target[i][0] + _target[i][1] * _256 + _target[i][2] * _256 * _256);
		}
		target.push_back(-1);
		sort(target.rbegin(), target.rend());

		int startImgX = srcImg.cols - 1;
		int startImgY = srcImg.rows - 1;
		int endImgX = 0;
		int endImgY = 0;

		for (int imgY = startNotWhiteY; imgY <= endNotWhiteY; imgY++) {
			int i = imgY * srcImg.cols + startNotWhiteX;
			for (int imgX = startNotWhiteX; imgX <= endNotWhiteX; imgX++) {
				auto srcPixel = srcData[i];

				bool isTarget = false;
				for (int i = 0; srcPixel <= target[i]; i++) {
					if (target[i] == srcPixel) {
						isTarget = true;
					}
				}

				if (isTarget) {
					dstData[i] = true;
					startImgX = min(startImgX, imgX);
					startImgY = min(startImgY, imgY);
					endImgX = max(endImgX, imgX);
					endImgY = max(endImgY, imgY);
				}
				i++;
			}
		}

		*_startImgX = startImgX;
		*_startImgY = startImgY;
		*_endImgX = endImgX;
		*_endImgY = endImgY;

		return dstImg;
	}

	Mat_<Vec3f> _apply(Mat_<Vec3f>& srcImg, const Mat_<bool>& targetFlagImg, int startImgX, int startImgY, int endImgX, int endImgY) {
		Mat_<Vec4f> dstImgY(srcImg.size());

		const int kernelSize = static_cast<int>(kernel.size());
		const int kernelCenter = kernelSize / 2;

		for (int imgY = startImgY; imgY <= endImgY; imgY++) {
			for (int imgX = startImgX; imgX < endImgX; imgX++) {
				if (targetFlagImg(imgY, imgX)) {
					Vec4f dstImgPixel(0, 0, 0, 0);
					for (int kernelIdx = 0; kernelIdx < kernelSize; kernelIdx++) {
						auto imgSampleX = clamp(imgX + kernelIdx - kernelCenter, 0, srcImg.cols - 1);
						if (targetFlagImg(imgY, imgSampleX)) {
							auto weight = kernel[kernelIdx];
							auto srcImgPixel = srcImg(imgY, imgSampleX);
							auto srcImgPixel_weighted = srcImgPixel * weight;
							dstImgPixel += Vec4f(srcImgPixel_weighted[0], srcImgPixel_weighted[1], srcImgPixel_weighted[2], weight);
						}
					}
					dstImgY(imgY, imgX) = dstImgPixel;
				}
			}
		}

		auto& dstImg = srcImg;

		for (int imgY = startImgY; imgY <= endImgY; imgY++) {
			for (int imgX = startImgX; imgX <= endImgX; imgX++) {
				if (targetFlagImg(imgY, imgX)) {
					Vec4f dstImgPixel(0, 0, 0);
					for (int kernelIdx = 0; kernelIdx < kernelSize; kernelIdx++) {
						auto imgSampleY = clamp(imgY + kernelIdx - kernelCenter, 0, srcImg.rows - 1);
						if (targetFlagImg(imgSampleY, imgX)) {
							auto weight = kernel[kernelIdx];
							auto srcImgPixel = dstImgY(imgSampleY, imgX);
							dstImgPixel += srcImgPixel * weight;
						}
					}
					dstImg(imgY, imgX) = *reinterpret_cast<Vec3f*>(&dstImgPixel) / dstImgPixel[3];
				}
			}
		}

		return dstImg;
	}

	Mat apply(Mat srcImg) {
		int startNotWhiteX, startNotWhiteY, endNotWhiteX, endNotWhiteY;
		auto iSrcImg = convertToIntImg(srcImg, &startNotWhiteX, &startNotWhiteY, &endNotWhiteX, &endNotWhiteY);

		Mat_<Vec3f> img = srcImg;


		for (auto target : targets) {
			int startImgX, startImgY, imgEndX, imgEndY;
			auto targetFlagImg = _createTargetFlagImg(iSrcImg, target, &startImgX, &startImgY, &imgEndX, &imgEndY, startNotWhiteX, startNotWhiteY, endNotWhiteX, endNotWhiteY);
			img = _apply(img, targetFlagImg, startImgX, startImgY, imgEndX, imgEndY);
		}

		Mat_<Vec3b> dstImg = img;
		return dstImg;
	}
};
