
/*
    Kaiqi Zhang
	Spring 2024
	CS 5330 Project 2

    Use color histogram as feature vector and chi square as distance metric
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

using namespace boost::filesystem;  

int main(int argc, char *argv[]) {
    const int histsize = 16;
    char csv_filename[256];
    strcpy(csv_filename, "../2dHistChiSquare.csv");

    //loop through each image file in a directory
    std::string directory_path = "../olympus";
    int num_images = 0;
    int total_images = 0;

    std::priority_queue<image_data_max> pq;

    cv::Mat target;
    char target_filename[256];
    strcpy( target_filename, argv[1]);
    target = cv::imread( target_filename );
    if (target.data == NULL) {
        printf("error: unable to read image %s\n", target_filename);
        return(-2);
    }
    //initialize the histogram for all images in database
    for (auto it : recursive_directory_range(directory_path)){ 
        total_images++;
        if (is_regular_file(it) && it.path().extension() == ".jpg") {
            num_images++;
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
            
            generateHist(image, imageHist, histsize);

            //prepare a vector to store the histogram data
            std::vector<float> hist_data_vec;

            for (int i = 0; i < histsize; i++) {
                for (int j = 0; j < histsize; j++) {
                    hist_data_vec.push_back(imageHist.at<float>(i, j));
                }
            }

            //append the histogram data to the csv file
            int result = append_image_data_csv(csv_filename, image_filename, hist_data_vec, 0);

        } 
    }
    printf("Number of files: %d\n", num_images);
    printf("Total number of files: %d\n", total_images);

    std::vector<char *> filenames;
    std::vector<std::vector<float>> histDataOfAllImages;
    std::vector<float> targetImageHist;

    read_image_data_csv(csv_filename, filenames, histDataOfAllImages, 0);
    printf("Number of files: %lu\n", filenames.size());


    int target_index = findTargetImageIndex(filenames, target_filename, histDataOfAllImages, targetImageHist);


    for (int i = 0; i < histDataOfAllImages.size(); ++i) {
        float chi_square = 0.0;
        for (int j = 0; j < histDataOfAllImages[i].size(); ++j) {
            float divisor = 1.0;
            if ((histDataOfAllImages[i][j] + targetImageHist[j]) != 0) {
                divisor = histDataOfAllImages[i][j] + targetImageHist[j];
            } 
            chi_square += (histDataOfAllImages[i][j] - targetImageHist[j]) * (histDataOfAllImages[i][j] - targetImageHist[j]) / divisor;
        }
        pq.push(image_data_max{chi_square, filenames[i]});
    }
    printf("Priority queue size: %zu\n", pq.size());


    image_data_max topResult[6];
    topResult[0] = pq.top();
    pq.pop();
    topResult[1] = pq.top();
    pq.pop();
    topResult[2] = pq.top();
    pq.pop();
    topResult[3] = pq.top();
    pq.pop();
    topResult[4] = pq.top();
    pq.pop();
    topResult[5] = pq.top();
    pq.pop();

    for (int i = 0; i < 6; ++i) {
        printf("Top result: %s, value: %f\n", topResult[i].filename.c_str(), topResult[i].size);
    }
    cv::namedWindow("Target", cv::WINDOW_AUTOSIZE);
    cv::Mat result1;
    cv::Mat result2;
    cv::Mat result3;
    cv::Mat result4;
    cv::Mat result5;
    cv::Mat result6;
    result1 = cv::imread(topResult[0].filename);
    result2 = cv::imread(topResult[1].filename);
    result3 = cv::imread(topResult[2].filename);
    result4 = cv::imread(topResult[3].filename);
    result5 = cv::imread(topResult[4].filename);
    result6 = cv::imread(topResult[5].filename);
    cv::imshow("Target", target);
    // cv::imshow("Result1", result1);
    cv::imshow("Result2", result2);
    cv::imshow("Result3", result3);
    cv::imshow("Result4", result4);
    cv::imshow("Result5", result5);
    cv::imshow("Result6", result6);

    cv::waitKey(0);
	cv::destroyWindow( "Target" );
	printf("Terminating\n");

    printf("\n");
    return 0;
}
