#ifndef UTIL_H
#define UTIL_H


#include <opencv2/opencv.hpp> 
#include <boost/filesystem.hpp>
using namespace boost::filesystem;  

int generateHist(cv::Mat &image, cv::Mat &imageHist,int histsize);
// int findTargetImageIndex(const std::vector<char *> &filenames, const char* target_filename);
int findTargetImageIndex(const std::vector<char *> filenames, const char * target_filename, 
    std::vector<std::vector<float>> histDataOfAllImages, std::vector<float> &targetImageHist);

int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );   
int createFilterKernel(std::vector<int>& vec1, std::vector<int>& vec2, cv::Mat &dst);
int applyLawsFilter(cv::Mat &src, cv::Mat &dst, cv::Mat &filter);
int normalizeImageWithL5L5(cv::Mat &normL5L5, cv::Mat &image);
int computelawsFilterHistogram(cv::Mat &grayImage, cv::Mat &filteredImage, cv::Mat &filter, std::vector<float> &histData);
int sobelX3x3Gray(cv::Mat &src, cv::Mat &dst);

struct recursive_directory_range
{
    typedef recursive_directory_iterator iterator;
    recursive_directory_range(path p) : p_(p) {}

    iterator begin() { return recursive_directory_iterator(p_); }
    iterator end() { return recursive_directory_iterator(); }

    path p_;
};

struct image_data_max {
    float size;
    std::string filename;

    bool operator<(const image_data_max& other) const {
        // Custom comparator for priority_queue
        if (size != other.size) {
            // Priority is based on the number, and in case of ties, use the string
            return size < other.size; // larger number has higher priority
        } else {
            return filename < other.filename;
        }
    }

};

struct image_data_min {
    double size;
    std::string filename;

    bool operator<(const image_data_min& other) const {
        // Custom comparator for priority_queue
        if (size != other.size) {
            // Priority is based on the number, and in case of ties, use the string
            return size > other.size; // smaller number has higher priority
        } else {
            return filename > other.filename;
        }
    }

};



#endif