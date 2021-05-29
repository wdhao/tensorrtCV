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
void testOnnx()
{
    string jsonPath = "D:/hrnet_ocr/test.json";
    trt *m_trt = new trt(jsonPath);
    m_trt->onnx2trt();
}
void dpV3PP()
{
    string jsonPath = "D:/qt_project/tensorrtF/model/deeplabV3_res50.json";
    trt *m_trt = new trt(jsonPath);
    m_trt->createENG();
}

int main()
{
    dpV3PP();
    //test();

//    float *s = new float[2];
//    s[0] = 1.1;
//    s[1] = 2.2;
//    cout<<*(static_cast<const float*>(s)+1)<<endl;
//    cout<<*(s+1)<<endl;
    cout << "Hello World!" << endl;
    return 0;
}
