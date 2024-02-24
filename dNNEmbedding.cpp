/*
    Kaiqi Zhang
	Spring 2024
	CS 5330 Project 2

    use feature vectors computed from ResNet18 deep network pre-trained on ImageNet
    use cosine similarity as the distance metric
*/
#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp> 
#include <boost/filesystem.hpp>
#include "csv_util/csv_util.h"
#include "util.h"

namespace fs = boost::filesystem;

int main(int argc, char *argv[]) {
    if(argc < 2) {
		printf("usage: %s < image target_filename>\n", argv[0]);
		exit(-1);
	}
    std::priority_queue<image_data_max> pq;
    std::string filePath = argv[1];
    fs::path p(filePath);
    std::string targetFilename = p.filename().string();
    char target_filename[256];
    std::strcpy( target_filename, targetFilename.c_str());
    target_filename[sizeof(target_filename) - 1] = 0;

    cv::Mat target;
    target = cv::imread( filePath );

    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    std::vector<float> targetData;

    read_image_data_csv("../ResNet18_olym.csv", filenames, data, 0);
    int target_index = findTargetImageIndex(filenames, target_filename, data, targetData);
    float targetDataNorm = 0.0;
    for (int i = 0; i < targetData.size(); i++) {
        targetDataNorm += targetData[i] * targetData[i];
    }
    targetDataNorm = sqrt(targetDataNorm);
    for (int i = 0; i < targetData.size(); i++) {
        targetData[i] /= targetDataNorm;
    }

    //compute cosine value between two vectors
    for (int i = 0; i < data.size(); i++) {
        float ImageNorm = 0.0;
        for (int j = 0; j < data[i].size(); j++) {
            ImageNorm += data[i][j] * data[i][j];
        }
        ImageNorm = sqrt(ImageNorm);
        float dotProduct = 0.0;

        for (int j = 0; j < data[i].size(); j++) {
            //normalize the feature vector
            data[i][j] /= ImageNorm;
            //take dot product of the two vectors
            dotProduct += data[i][j] * targetData[j];
        }
        pq.push(image_data_max{dotProduct, filenames[i]});
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
    return 0;
}
