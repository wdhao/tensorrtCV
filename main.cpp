#include "trt.h"
#include "utils.h"

void test()
{
    string jsonPath = "D:/hrnet_ocr/test.json";
    trt *m_trt = new trt(jsonPath);
    m_trt->createENG();
    int batchsize = 1;
    m_trt->inference_init(batchsize);
    float *a = new float[batchsize*100];
    for(int i = 0; i < batchsize*100; i++)
        a[i] = i*1.0 +1;
    float *out = new float[batchsize*56];
    m_trt->doInference(a,batchsize,out);
    for(int i = 0;i<56;i++)
    {
        cout<<out[i]<<" ";
        if(i%8 == 7)
            cout<<endl;
    }
}
void resnet(string JsonPath)
{
    trt *m_trt = new trt(JsonPath);
    m_trt->createENG();
}

int main(int argc ,char** argv)
{
    string JsonPath = argv[1];
    trt *m_trt = new trt(JsonPath);
    m_trt->createENG();
    int batchsize = 1;
    m_trt->inference_init(batchsize);
    float *input = new float[batchsize * m_trt->param.input_c * m_trt->param.input_h * m_trt->param.input_w];
    float *output = new float[batchsize * m_trt->param.outputSize];
    m_trt->doInference(input,1,output);
    return 0;
}
