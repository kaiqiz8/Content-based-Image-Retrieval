/*
    Kaiqi Zhang
	Spring 2024
	CS 5330 Project 2

    use laws filter histogram as feature vector and squared difference as the distance metric
*/

#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp> 
#include <boost/filesystem.hpp>
#include "csv_util/csv_util.h"
#include "util.h"

int main(int argc, char *argv[]) {
    char target_filename[256];
    strcpy( target_filename, argv[1]);
    cv::Mat target;
    target = cv::imread( target_filename );
    if (target.data == NULL) {
        printf("error: unable to read image %s\n", target_filename);
        return(-2);
    }

    char lawsFilter_filename[256];
    strcpy(lawsFilter_filename, "../lawsFilter.csv");
    std::string directory_path = "../olympus";
    std::priority_queue<image_data_min> pq;

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

            // L5 (Level): [1, 4, 6, 4, 1]
            // E5 (Edge): [-1, -2, 0, 2, 1]
            // S5 (Spot): [-1, 0, 2, 0, -1]
            // W5 (Wave): [-1, 2, 0, -2, 1]
            // R5 (Ripple): [1, -4, 6, -4, 1]
            cv::Mat L5 = (cv::Mat_<float>(1,5) << 1, 4, 6, 4, 1);
            cv::Mat E5 = (cv::Mat_<float>(1,5) << -1, -2, 0, 2, 1);
            cv::Mat S5 = (cv::Mat_<float>(1,5) << -1, 0, 2, 0, -1);
            cv::Mat W5 = (cv::Mat_<float>(1,5) << -1, 2, 0, -2, 1);
            cv::Mat R5 = (cv::Mat_<float>(1,5) << 1, -4, 6, -4, 1);

            std::vector<cv::Mat> matList;
            matList.push_back(L5);
            matList.push_back(E5);
            matList.push_back(S5);
            matList.push_back(W5);
            matList.push_back(R5);

            std::vector<float> histData;
            //loop through all unique combinations of the 5 filters
            for (int i = 0; i < matList.size(); i++) {
                for (int j = i; j < matList.size(); j++) {
                    cv::Mat filter = matList[i].t() * matList[j];
                    cv::Mat filteredImage;
                    computelawsFilterHistogram(grayImage, filteredImage, filter, histData);
                }
            }
            append_image_data_csv(lawsFilter_filename, image_filename, histData, 0);
        }
    }
    
    std::vector<char *> filenames;
    std::vector<std::vector<float>> histDataOfAllImages;
    std::vector<float> targetImageHist;

    read_image_data_csv(lawsFilter_filename, filenames, histDataOfAllImages);
    findTargetImageIndex(filenames, target_filename, histDataOfAllImages, targetImageHist);
    for (int i = 0; i < histDataOfAllImages.size(); i++) {
        float sum = 0.0;
        for (int j = 0; j < histDataOfAllImages[i].size(); j++) {
            sum += (targetImageHist[j] - histDataOfAllImages[i][j]) 
                * (targetImageHist[j] - histDataOfAllImages[i][j]);
        }
        pq.push(image_data_min{sum, filenames[i]});
    }


    image_data_min topResult[4];
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