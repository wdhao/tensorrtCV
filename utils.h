#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace std;

#ifndef CUDA_CHECK

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#endif

template<typename T>
void write(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template<typename T>
void read(const char*& buffer, T& val)
{
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}


void mblobFromImages(cv::InputArrayOfArrays images_, cv::OutputArray blob_,
    cv::Size size, const cv::Scalar& mean_, const cv::Scalar& std_, bool swapRB, bool crop);
cv::Mat BlobFromImages(cv::InputArrayOfArrays images, cv::Size size,
    const cv::Scalar& mean, const cv::Scalar& std_num, bool swapRB, bool crop);
#endif // UTILS_H
