/*
	Kaiqi Zhang
	Spring 2024
	CS 5330 Project 2

    Use a single normalized 2D histogram as a feature vector. 
    Use histogram intersection as the distance metric.
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
    float max;
    char csv_filename[256];
    std::priority_queue<image_data_max> pq; // priority queue
    strcpy(csv_filename, "../2dHist.csv");
    if (argc < 2) {
        printf("usage: %s <target image>\n", argv[0]);
        return(-1);
    }

    cv::Mat target;
    char target_filename[256];
    strcpy( target_filename, argv[1]);
    target = cv::imread( target_filename );
    if (target.data == NULL) {
        printf("error: unable to read image %s\n", target_filename);
        return(-2);
    }
    cv::Mat targetHist;
    targetHist = cv::Mat::zeros(cv::Size(histsize, histsize), CV_32F);
    for (int i = 0; i < target.rows; i++) {
        cv::Vec3b *ptr = target.ptr<cv::Vec3b>(i); // pointer to the i-th row
        for (int j = 0; j < target.cols; j++) {
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
            targetHist.at<float>(rindex, gindex)++;
            
            //keep track of the size of the largest bucket
            float newValue = targetHist.at<float>(rindex, gindex);
            max = newValue > max ? newValue : max;
        }
    }
    targetHist /= (target.rows * target.cols);
    std::vector<float> target_hist_data_vec;
    for (int i = 0; i < histsize; i++) {
        for (int j = 0; j < histsize; j++) {
            target_hist_data_vec.push_back(targetHist.at<float>(i, j));
        }
    }

    //loop through each image file in a directory
    std::string directory_path = "../olympus";
    int num_images = 0;
    int total_images = 0;
    for (auto it : recursive_directory_range(directory_path)){ 
        total_images++;
        if (is_regular_file(it) && it.path().extension() == ".jpg") {
            num_images++;
            // printf(it.path().string().c_str());
            cv::Mat imageHist;
            cv::Mat image;
            char image_filename[256];
            strcpy( image_filename, it.path().string().c_str());
            image = cv::imread( image_filename );
            if (image.data == NULL) {
                printf("error: unable to read image %s\n", image_filename);
                return(-2);
            }
            //initialize the histogram
            imageHist = cv::Mat::zeros(cv::Size(histsize, histsize), CV_32F);
            
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
                    max = newValue > max ? newValue : max;
                }
            }
            imageHist /= (image.rows * image.cols); //divide all the elements of the histogram by the number of pixels
    
            std::vector<float> hist_data_vec;
            float sum_min = 0.0;
            for (int i = 0; i < histsize; i++) {
                for (int j = 0; j < histsize; j++) {
                    hist_data_vec.push_back(imageHist.at<float>(i, j));
                    sum_min += std::min(imageHist.at<float>(i, j), targetHist.at<float>(i, j));
                }
            }
            pq.push(image_data_max{sum_min, it.path().string()});

            int result = append_image_data_csv(csv_filename, image_filename, hist_data_vec, 0);
            
        } 
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

