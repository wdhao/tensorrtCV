#include <iterator>
#include "calibrator.h"
#include "utils.h"


vector<string> loadImages(const string imgTxt)
{
    vector<string> imgInfo;
    FILE *f = fopen(imgTxt.c_str(),"r");
    if (!f){
         perror("Error");
        cout<<"cant open file"<<imgTxt<<endl;
        return imgInfo;
    }
    char str[512];
    while (fgets(str,512,f)!=NULL)
    {
        for (int i = 0;str[i] != '\0';++i) {
            if (str[i] == '\r')
            {
                str[i] = '\0';
            }
            if (str[i] == '\n')
            {
                str[i] = '\0';
                break;
            }
        }
        imgInfo.push_back(str);
    }
    fclose(f);
    return imgInfo;
}


calibrator::calibrator(const unsigned int &batchsize,
                       const string &caliTxt,
                       const string &calibratorPath,
                       const unsigned int &inputC,
                       const unsigned int &inputH,
                       const unsigned int &inputW,
                       const string &inputName,
                       vector<float> Mean,
                       vector<float> Std,
                       bool isDiv255):m_batchsize(batchsize),
                                       m_inputC(inputC),
                                       m_inputH(inputH),
                                       m_inputW(inputW),                                       
                                       m_InputCount(batchsize * inputC * inputH * inputW),
                                       m_inputName(inputName.c_str()),
                                       m_calibratorPath(calibratorPath),
                                       m_ImageIndex(0)
{
    m_ImageList = loadImages(caliTxt);
    m_mean = Mean;
    m_std = Std;

    div_255 = isDiv255;

    cudaMalloc(&m_CudaInput,m_InputCount*sizeof (float));

}
int calibrator::getBatchSize() const
{
    return m_batchsize;
}
bool calibrator::getBatch(void **bindings, const char **names, int nbBindings)
{

    if(m_ImageIndex + m_batchsize > m_ImageList.size()){
        return false;
    }
    int flag = 0;
    cv::Scalar Mean;
    cv::Scalar Std;

    if(m_inputC == 3)
    {
        flag = 1;
        Mean = cv::Scalar(m_mean[0], m_mean[1], m_mean[2]);
        Std = cv::Scalar(m_std[0], m_std[1], m_std[2]);
    }
    else if(m_inputC == 1){
        Mean = cv::Scalar(m_mean[0]);
        Std = cv::Scalar(m_std[0]);
    }
    else {
        cout<<"not support "<<m_inputC<<" channels"<<endl;
    }
    vector<cv::Mat> InputImgs;
    for (unsigned int i = m_ImageIndex; i < m_ImageIndex + m_batchsize;i++) {
        string imgPath = m_ImageList.at(i);
        cout<<imgPath<<endl;
        cv::Mat temp = cv::imread(imgPath,flag);
        if(temp.empty()){
            cout<<imgPath<<" is not a image!"<<endl;
        }
        cv::Mat img ;
        if(div_255)
        {
            temp.convertTo(img,CV_32FC1,1.0/255.0);
            InputImgs.push_back(img);
        }
        else {
            InputImgs.push_back(temp);
        }
    }
    m_ImageIndex += m_batchsize;
    cv::Mat trtInput = BlobFromImages(InputImgs,cv::Size(m_inputH,m_inputW),
                                       Mean,Std,
                                       true,false);
//    cv::Mat trtInput = cv::dnn::blobFromImages(InputImgs,1.0,cv::Size(m_inputH,m_inputW),Mean,true,false);
    cudaMemcpy(m_CudaInput,trtInput.ptr<float>(0),m_InputCount*sizeof (float),cudaMemcpyHostToDevice);

    bindings[0] = m_CudaInput;
    return true;
}
const void *calibrator::readCalibrationCache(size_t &length)
{
    void *output;
    m_CalibrationCache.clear();
    ifstream input(m_calibratorPath,ios::binary);
    input >> noskipws;
    if (input.good())
    {
        copy(istream_iterator<char>(input),istream_iterator<char>(),back_inserter(m_CalibrationCache));
    }
    length = m_CalibrationCache.size();

    if(length){
        std::cout << "Using cached calibration table to build the engine " << std::endl;
        output = &m_CalibrationCache[0];
    }
    else {
        std::cout << "New calibration table will be created to build the engine" << std::endl;
        output = nullptr;
    }

    return output;
}
void calibrator::writeCalibrationCache(const void *ptr, std::size_t length)
{
    assert(!m_calibratorPath.empty());
    cout<<"length =  "<<length<<endl;
    ofstream output(m_calibratorPath,ios::binary);
    output.write(reinterpret_cast<const char*>(ptr),length);
    output.close();
}
