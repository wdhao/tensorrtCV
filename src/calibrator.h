#pragma once
#ifndef CALIBRATOR_H
#define CALIBRATOR_H
#include <fstream>
#include <NvInfer.h>
#include <cuda.h>
#include <string>
#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>

using namespace std;
using namespace nvinfer1;

class calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    calibrator(const unsigned int &batchsize,
               const string &caliTxt,
               const string &calibratorPath,
               const unsigned int &inputC,
               const unsigned int &inputH,
               const unsigned int &inputW,
               const string &inputName,
               vector<float> Mean,
               vector<float> Std,
               bool isDiv255);

    int getBatchSize() const override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
    const void* readCalibrationCache(size_t& length) override;
    void writeCalibrationCache(const void* ptr, std::size_t length) override;

private:
    unsigned int m_batchsize;
    const unsigned int m_inputC;
    const unsigned int m_inputH;
    const unsigned int m_inputW;
    vector<float> m_mean;
    vector<float> m_std;
    //const uint64_t m_inputSize;
    const uint64_t m_InputCount;
    const char *m_inputName;
    const string m_calibratorPath{nullptr};
    vector<string> m_ImageList;
    void *m_CudaInput{nullptr};
    vector<char> m_CalibrationCache;
    unsigned int m_ImageIndex;
    bool div_255;

};

#endif // CALIBRATOR_H
