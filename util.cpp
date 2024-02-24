#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp> 

int generateHist(cv::Mat &image, cv::Mat &imageHist,int histsize) {
    for (int i = 0; i < image.rows; i++) {
        cv::Vec3b *ptr = image.ptr<cv::Vec3b>(i); // pointer to the i-th row
        for (int j = 0; j < image.cols; j++) {
            //get the RGB values 
            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];

            //compute the r, g chromaticity
            float divisor = (R + G + B);
            divisor = divisor > 0.0 ? divisor : 1.0; // check for zero
            float r = R / divisor;
            float g = G / divisor;

            //compute the histogram index
            int rindex = (int)(r * (histsize - 1) + 0.5);
            int gindex = (int)(g * (histsize - 1) + 0.5);
            imageHist.at<float>(rindex, gindex)++;
            
            //keep track of the size of the largest bucket
            float newValue = imageHist.at<float>(rindex, gindex);
            // max = newValue > max ? newValue : max;
        }
    }
    imageHist /= (image.rows * image.cols);
    return 0;
            
}

int generateGradientHist() {
    
}

int findTargetImageIndex(const std::vector<char *> filenames, const char * target_filename,
    std::vector<std::vector<float>> histDataOfAllImages, std::vector<float> &targetImageHist) {
    int target_index = 0;
    for (int i = 0; i < filenames.size(); i++) {
        // printf("Filename: %s\n", filenames[i]);
        if (strcmp(filenames[i], target_filename) == 0) {
            target_index = i;
            targetImageHist = histDataOfAllImages[i];
            printf("Found Target image");
            break;
        }
    }
    return target_index;

}

int sobelX3x3( cv::Mat &src, cv::Mat &dst ) {
    src.copyTo(dst);

    for (int i = 1; i < src.rows - 1; i++){
        for (int j = 1; j < src.cols - 1; j++){
            for (int k = 0; k < src.channels(); k++) {
                int sum = (-1) * src.at<cv::Vec3b>(i-1, j-1)[k] + src.at<cv::Vec3b>(i-1, j+1)[k] 
                        + (-2) * src.at<cv::Vec3b>(i, j-1)[k] + 2 * src.at<cv::Vec3b>(i, j+1)[k]
                        + (-1) * src.at<cv::Vec3b>(i+1, j-1)[k] + src.at<cv::Vec3b>(i+1, j+1)[k];

                dst.at<cv::Vec3b>(i,j)[k] = static_cast<short>(sum);
            }
        }
    }
    return(0);
}

int sobelX3x3Gray( cv::Mat &src, cv::Mat &dst ) {
    src.copyTo(dst);

    for (int i = 1; i < src.rows - 1; i++){
        for (int j = 1; j < src.cols - 1; j++){
            int sum = (-1) * src.at<uchar>(i-1, j-1) + src.at<uchar>(i-1, j+1) 
                    + (-2) * src.at<uchar>(i, j-1) + 2 * src.at<uchar>(i, j+1)
                    + (-1) * src.at<uchar>(i+1, j-1) + src.at<uchar>(i+1, j+1);
        }
    }
    return(0);
}


int sobelY3x3( cv::Mat &src, cv::Mat &dst ) {
    src.copyTo(dst);

    for (int i = 1; i < src.rows - 1; i++){
        for (int j = 1; j < src.cols - 1; j++){
            for (int k = 0; k < src.channels(); k++) {
                int sum = src.at<cv::Vec3b>(i-1, j-1)[k] + 2 * src.at<cv::Vec3b>(i-1, j)[k] + src.at<cv::Vec3b>(i-1, j+1)[k] 
                        + (-1) * src.at<cv::Vec3b>(i+1, j-1)[k]  + (-2) * src.at<cv::Vec3b>(i+1, j)[k] + (-1) * src.at<cv::Vec3b>(i+1, j+1)[k];

                dst.at<cv::Vec3b>(i,j)[k] = static_cast<short>(sum);
            }
        }
    }
    return(0);
}

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){
    sx.copyTo(dst);
    for (int i = 0; i < sx.rows; i++){
        for (int j = 0; j < sx.cols; j++){
            for (int k = 0; k < sx.channels(); k++) {
                float value1 = static_cast<float>(sx.at<cv::Vec3s>(i, j)[k]);
                float value2 = static_cast<float>(sy.at<cv::Vec3s>(i, j)[k]);
                dst.at<cv::Vec3s>(i, j)[k] = std::sqrt(value1 * value1 + value2 * value2);
            }   
        }        
    }
    return (0);
}

int createFilterKernel(std::vector<int>& vec1, std::vector<int>& vec2, cv::Mat &dst) {
    for (int i = 0; i < vec1.size(); i++) {
        for (int j = 0; j < vec2.size(); j++) {
            dst.at<float>(i, j) = vec1[i] * vec2[j];
        }
    }
    return 0;
}

int applyLawsFilter(cv::Mat &src, cv::Mat &dst, cv::Mat &filter) {
    src.copyTo(dst);
    for (int i = 2; i < src.rows - 2; i++){
        for (int j = 2; j < src.cols - 2; j++){
            // Extract a 5x5 matrix from src image
            cv::Mat roi = src(cv::Range(i - 2, i + 3), cv::Range(j - 2, j + 3));
            cv::filter2D(roi, dst, -1, filter);
        }
    }
    for (int i = 2; i < src.rows - 2; i++){
        for (int j = 2; j < src.cols - 2; j++){
            dst.at<float>(i, j) = dst.at<float>(i, j) / 256.0;
        }
    }
    return 0;
}

int normalizeImageWithL5L5(cv::Mat &normL5L5, cv::Mat &image) {
    // src.copyTo(dst);
    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            image.at<uchar>(i, j) = image.at<uchar>(i, j) / normL5L5.at<uchar>(i, j);
        }
    }
    return 0;
}

int computelawsFilterHistogram(cv::Mat &grayImage, cv::Mat &filteredImage, cv::Mat &filter, std::vector<float> &histData) {
    cv::filter2D(grayImage, filteredImage, CV_32F, filter);

    // cv::Mat textureEnergy;
    cv::convertScaleAbs(filteredImage, filteredImage);

    int histSize = 8; // or 8 for a coarse histogram
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&filteredImage, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    // Normalize the histogram if desired
    cv::normalize(hist, hist, 0, 256, cv::NORM_MINMAX);

    for (int i = 0; i < hist.rows; i++) {
        histData.push_back(hist.at<float>(0, i));
    }

    return 0;
}