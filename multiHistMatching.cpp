/*
    Kaiqi Zhang
	Spring 2024
	CS 5330 Project 2

    center 9x9 matrix and whole image color histogram as a feature vector.
    distance matrix is the sum of histogram intersection.
*/

#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp> 
#include <boost/filesystem.hpp>
#include "csv_util/csv_util.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "util.h"

//generate 9x9 matrix from the center of the image
int getCenter9X9Matrix(cv::Mat &src, cv::Mat &dst) {
    int filterSize = 19;
    int center_rol = src.size().height / 2;
    int center_col = src.size().width / 2;
    cv::Rect roi(center_col - filterSize/2, center_rol - filterSize/2, filterSize, filterSize);
    dst = src(roi);
    return 0;
}

int main(int argc, char *argv[]) {
    const int histsize = 16;
    char csv_filename[256];
    char csv_full_image_filename[256];
    strcpy(csv_filename, "../2dCenterHist.csv");
    strcpy(csv_full_image_filename, "../2dFullImageHist.csv");

    std::string directory_path = "../olympus";
    std::priority_queue<image_data_max> pq;
    std::map<std::string, float> centerImageDictionary;

    char target_filename[256];
    strcpy( target_filename, argv[1]);
    cv::Mat target;
    target = cv::imread( target_filename );
    if (target.data == NULL) {
        printf("error: unable to read image %s\n", target_filename);
        return(-2);
    }

    for (auto it : recursive_directory_range(directory_path)){ 
        if (is_regular_file(it) && it.path().extension() == ".jpg") {
            cv::Mat image;
            cv::Mat imageCenter;
            cv::Mat imageCenterHist;

            cv::Mat imageFullHist;

            char image_filename[256];
            strcpy( image_filename, it.path().string().c_str());
            image = cv::imread( image_filename );
            if (image.data == NULL) {
                printf("error: unable to read image %s\n", image_filename);
                return(-3);
            }
            getCenter9X9Matrix(image, imageCenter);
            imageCenterHist = cv::Mat::zeros(cv::Size(histsize, histsize), CV_32F);
            imageFullHist = cv::Mat::zeros(cv::Size(histsize, histsize), CV_32F);

            for (int i = 0; i < imageCenter.rows; i++) {
                cv::Vec3b *ptr = imageCenter.ptr<cv::Vec3b>(i);
                for (int j = 0; j < imageCenter.cols; j++) {
                    float B = ptr[j][0];
                    float G = ptr[j][1];
                    float R = ptr[j][2];

                    float divisor = R + G + B;
                    divisor = divisor > 0.0 ? divisor : 1.0;
                    float r = R / divisor;
                    float g = G / divisor;

                    int rindex = (int)(r * (histsize - 1) + 0.5);
                    int gindex = (int)(g * (histsize - 1) + 0.5);
                    imageCenterHist.at<float>(rindex, gindex)++;
                    
                }
            }

            for (int i = 0; i < image.rows; i++){
                cv::Vec3b *ptrF = image.ptr<cv::Vec3b>(i);
                for (int j = 0; j < image.cols; j++) {
                    float BF = ptrF[j][0];
                    float GF = ptrF[j][1];
                    float RF = ptrF[j][2];

                    float divisorF = RF + GF + BF;
                    divisorF = divisorF > 0.0 ? divisorF : 1.0;
                    float rF = RF / divisorF;
                    float gF = GF / divisorF;

                    int rindexF = (int)(rF * (histsize - 1) + 0.5);
                    int gindexF = (int)(gF * (histsize - 1) + 0.5);
                    imageFullHist.at<float>(rindexF, gindexF)++;
                }
            }

            imageCenterHist /= (imageCenter.rows * imageCenter.cols);
            imageFullHist /= (image.rows * image.cols);

            std::vector<float> hist_data_vec;
            std::vector<float> fullImageHistDataVec;
            for (int i = 0; i < histsize; i++) {
                for (int j = 0; j < histsize; j++) {
                    hist_data_vec.push_back(imageCenterHist.at<float>(i, j));
                    fullImageHistDataVec.push_back(imageFullHist.at<float>(i, j));
                }
            }
            
            append_image_data_csv(csv_filename, image_filename, hist_data_vec, 0);
            append_image_data_csv(csv_full_image_filename, image_filename, fullImageHistDataVec, 0);
        }
    }

    int target_index = 0;
    std::vector<char *> filenames;
    std::vector<std::vector<float>> centerHistDataOfAllImages;
    std::vector<float> centerTargetImageHist;

    std::vector<std::vector<float>> fullHistDataOfAllImages;
    std::vector<float> fullImageTargetImageHist;
    read_image_data_csv(csv_full_image_filename, filenames, fullHistDataOfAllImages, 0);

    read_image_data_csv(csv_filename, filenames, centerHistDataOfAllImages, 0);
    for (int i = 0; i < filenames.size(); i++) {
        if (strcmp(filenames[i], target_filename) == 0) {
            target_index = i;
            centerTargetImageHist = centerHistDataOfAllImages[i];
            fullImageTargetImageHist = fullHistDataOfAllImages[i];
            break;
        }
    }

// calculate the intersection of the histogram
    for (int i = 0; i < centerHistDataOfAllImages.size(); i++) {
        float sum = 0.0;
        for (int j = 0; j < centerTargetImageHist.size(); j++) {
            sum += std::min(centerTargetImageHist[j], centerHistDataOfAllImages[i][j]);
        }
        centerImageDictionary[filenames[i]] = sum;
        
    }   

    for (int i = 0; i < fullHistDataOfAllImages.size(); i++) {
        float sum = 0.0;
        for (int j = 0; j < fullImageTargetImageHist.size(); j++) {
            sum += std::min(fullImageTargetImageHist[j], fullHistDataOfAllImages[i][j]);
        }
        pq.push(image_data_max{centerImageDictionary[filenames[i]] + sum, filenames[i]});
    }



    image_data_max topResult[4];
    topResult[0] = pq.top();
    pq.pop();
    topResult[1] = pq.top();
    pq.pop();
    topResult[2] = pq.top();
    pq.pop();
    topResult[3] = pq.top();
    pq.pop();

    for (int i = 0; i < 4; ++i) {
        printf("Top result: %s, value: %f\n", topResult[i].filename.c_str(), topResult[i].size);
    }
    cv::namedWindow("Target", cv::WINDOW_AUTOSIZE);
    cv::Mat result1;
    cv::Mat result2;
    cv::Mat result3;
    cv::Mat result4;
    result1 = cv::imread(topResult[0].filename);
    result2 = cv::imread(topResult[1].filename);
    result3 = cv::imread(topResult[2].filename);
    result4 = cv::imread(topResult[3].filename);
    cv::imshow("Target", target);
    cv::imshow("Result2", result2);
    cv::imshow("Result3", result3);
    cv::imshow("Result4", result4);

    cv::waitKey(0);
    cv::destroyWindow( "Target" );
    printf("Terminating\n");

    return 0;
}

