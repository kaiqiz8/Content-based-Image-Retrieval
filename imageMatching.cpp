/*
	Kaiqi Zhang
	Spring 2024
	CS 5330 Project 2

    Use the 7x7 square in the middle of the image as a feature vector.
    Use sum-of-squared-difference as the distance metric to calculate across three color channels. 
*/

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <queue>
#include <opencv2/opencv.hpp> 
#include <boost/filesystem.hpp>
#include "util.h"

//function to calculate the sum of squared differences between the target and the match image
double getImageFeatures(cv::Mat &target, cv::Mat &match) {
    int filterSize = 7;

    int target_center_rol = target.size().height / 2;
    int target_center_col = target.size().width / 2;

    int match_center_rol = target.size().height / 2;
    int match_center_col = target.size().width / 2;

    cv::Rect roi(target_center_col - filterSize/2, target_center_rol - filterSize/2, filterSize, filterSize);
    cv::Rect roi2(match_center_col - filterSize/2, match_center_rol - filterSize/2, filterSize, filterSize);

    cv::Mat target_center_region = target(roi);
    cv::Mat match_center_region = match(roi2);
    double difference = 0.0;
    for (int i = 0; i < 7; i++ ){
        for (int j = 0; j < 7; j++) {
            for (int channel = 0; channel < 3; channel++) {
                difference += (target_center_region.at<cv::Vec3b>(i, j)[channel] - match_center_region.at<cv::Vec3b>(i, j)[channel]) 
                            * (target_center_region.at<cv::Vec3b>(i, j)[channel] - match_center_region.at<cv::Vec3b>(i, j)[channel]);
            }
        }
    }
    return difference;
}


int main(int argc, char *argv[]) {
    std::string directory_path = "../olympus";
    cv::Mat image;
    cv::Mat target;
    cv::Mat result1;
    cv::Mat result2;
    cv::Mat result3;

    if(argc < 2) {
		printf("usage: %s < image target_filename>\n", argv[0]);
		exit(-1);
	}

    char target_filename[256];
    strcpy( target_filename, argv[1]);
    target = cv::imread( target_filename );
    int counter = 0;
    std::priority_queue<image_data_min> pq; 
    
    //iterate through all the images in the directory and calculate the sum of squared differences
    for (auto it : recursive_directory_range(directory_path)){ 
        if (is_regular_file(it) && it.path().extension() == ".jpg") {
            counter++;
            image = cv::imread(it.path().string());
            if (image.data == NULL) { // no image data read from file
                exit(-1);
            }
            double image_feature_data = getImageFeatures(target, image);
            pq.push(image_data_min{image_feature_data, it.path().string()});
        }
    }
    printf("Number of files: %d\n", counter);
    printf("Priority queue size: %zu\n", pq.size());
    image_data_min topResult[4];

//set up priority queue to order the images by the smallest sum of squared differences, the smallest sum will be on top
    std::priority_queue<image_data_min> temp; 
    for (int i = 0; i < 4 && !pq.empty(); ++i) {
        topResult[i] = pq.top();
        pq.pop();
    }

    for (int i = 0; i < 4; ++i) {
        printf("Top result: %s, value: %f\n", topResult[i].filename.c_str(), topResult[i].size);
    }

    result1 = cv::imread(topResult[1].filename);
    result2 = cv::imread(topResult[2].filename);
    result3 = cv::imread(topResult[3].filename);

    cv::namedWindow("Target", cv::WINDOW_AUTOSIZE);
    cv::imshow("Target", target);
    cv::imshow("Result1", result1);
    cv::imshow("Result2", result2);
    cv::imshow("Result3", result3);

    cv::waitKey(0);
	cv::destroyWindow( "Target" );
	printf("Terminating\n");

    return(0);
}