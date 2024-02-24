/*
    Kaiqi Zhang
	Spring 2024
	CS 5330 Project 2

    Compute texture metric by calculating the gradient magnitude histogram. 
    Distance metric is the sum of histogram intersection.
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

int main(int argc, char *argv[]) {
    const int histsize = 16;
    char csv_texture_filename[256];
    char csv_full_image_filename[256];
    strcpy(csv_texture_filename, "../2dTexture.csv");
    strcpy(csv_full_image_filename, "../2dFullImageTexture.csv");

    std::string directory_path = "../olympus";
    std::priority_queue<image_data_max> pq;
    std::map<std::string, float> textureHistDictionary;

    char target_filename[256];
    strcpy( target_filename, argv[1]);
    cv::Mat target;
    target = cv::imread( target_filename );
    if (target.data == NULL) {
        printf("error: unable to read image %s\n", target_filename);
        return(-2);
    }

    for (auto it: recursive_directory_range(directory_path)) {
        if (is_regular_file(it) && it.path().extension() == ".jpg") {
            cv::Mat image;
            char image_filename[256];
            strcpy( image_filename, it.path().string().c_str());
            image = cv::imread( image_filename );
            if (image.data == NULL) {
                printf("error: unable to read image %s\n", image_filename);
                return(-2);
            }
            cv::Mat grayImage;
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

            cv::Mat grad_x, grad_y;
            cv::Mat abs_grad_x, abs_grad_y;
            cv::Mat magnitude;
            
            cv::Sobel(grayImage, grad_x, CV_64F, 1, 0, 3);
            cv::Sobel(grayImage, grad_y, CV_64F, 0, 1, 3);

            cv::convertScaleAbs(grad_x, abs_grad_x);
            cv::convertScaleAbs(grad_y, abs_grad_y);
            
            cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, magnitude);

            cv::Mat sobelGradientHist;
            cv::Mat imageHist;
            imageHist = cv::Mat::zeros(cv::Size(histsize, histsize), CV_32F);
            int gradientHistSize = 256;
            float range[] = {0, 256};
            const float *histRange = {range};
            cv::calcHist(&magnitude, 1, 0, cv::Mat(), sobelGradientHist, 1, &gradientHistSize, &histRange, true, false);
            cv::normalize(sobelGradientHist, sobelGradientHist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

            generateHist(image, imageHist, histsize);

            std::vector<float> hist_sobel_data_vec;
            std::vector<float> fullImageHistDataVec;
            for (int i = 0; i < histsize; i++) {
                hist_sobel_data_vec.push_back(sobelGradientHist.at<float>(0, i));
                for (int j = 0; j < histsize; j++) {
                    fullImageHistDataVec.push_back(imageHist.at<float>(i, j));
                }
            }
            
            append_image_data_csv(csv_texture_filename, image_filename, hist_sobel_data_vec, 0);
            append_image_data_csv(csv_full_image_filename, image_filename, fullImageHistDataVec, 0);
        }

    }
    std::vector<char *> filenames;

    std::vector<std::vector<float>> sobelHistDataOfAllImages;
    std::vector<float> sobelTargetImageHist;

    std::vector<std::vector<float>> fullHistDataOfAllImages;
    std::vector<float> fullImageTargetImageHist;
    read_image_data_csv(csv_texture_filename, filenames, sobelHistDataOfAllImages);
    read_image_data_csv(csv_full_image_filename, filenames, fullHistDataOfAllImages);

    findTargetImageIndex(filenames, target_filename, sobelHistDataOfAllImages, sobelTargetImageHist);
    findTargetImageIndex(filenames, target_filename, fullHistDataOfAllImages, fullImageTargetImageHist);

    for (int i = 0; i < sobelHistDataOfAllImages.size(); ++i) {
        float sum_min = 0.0;
        for (int j = 0; j < sobelHistDataOfAllImages[i].size(); ++j) {
            sum_min += std::min(sobelHistDataOfAllImages[i][j], sobelTargetImageHist[j]);
        }
        textureHistDictionary[filenames[i]] = sum_min;
    }

    for (int i = 0; i < fullHistDataOfAllImages.size(); ++i) {
        float sum_min = 0.0;
        for (int j = 0; j < fullHistDataOfAllImages[i].size(); ++j) {
            sum_min += std::min(fullHistDataOfAllImages[i][j], fullImageTargetImageHist[j]);
        }
        pq.push(image_data_max{textureHistDictionary[filenames[i]] + sum_min, filenames[i]});
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
    // cv::imshow("Result1", result1);
    cv::imshow("Result2", result2);
    cv::imshow("Result3", result3);
    cv::imshow("Result4", result4);

    cv::waitKey(0);
    cv::destroyWindow( "Target" );
    printf("Terminating\n");



    return 0;


}

