#include "trt.h"
#include "utils.h"
#include "opencv2/opencv.hpp"

cv::Mat createLTU(int len) {
	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.data;
	for (int j = 0; j < 256; ++j) {
		p[j] = (j * (256 / len) > 255) ? uchar(255) : (uchar)(j * (256 / len));
	}
	return lookUpTable;
}

void test()
{
    string jsonPath = "D:/hrnet_ocr/test.json";
    trt *m_trt = new trt(jsonPath);
    m_trt->createENG();
    int batchsize = 1;
    m_trt->inference_init(batchsize, m_trt->getOutDim());
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

void HRnet(std::string jsonPath)
{
	//string jsonPath = "/mnt/f/LearningCode/ddrnet/ddrnet-slim2-wsl_infer.json";
	trt *m_trt = new trt(jsonPath);
	if(m_trt->param.createENG)
	{
		m_trt->createENG();
		m_trt->param.outputSize = m_trt->getOutDim();
		std::cout <<"Create engine..."<< std::endl;
		std::cout <<"outputSize=" << m_trt->getOutDim() << std::endl;
		// write json
		 Json::Reader m_Reader;
    	Json::Value root;
    	ifstream fp;
    	fp.open(jsonPath,ios::binary);
    	m_Reader.parse(fp,root);

		Json::StyledWriter writer;
		root["outputSize"] = m_trt->getOutDim();
		root["createENG"] = false;
		ofstream os;
		auto replacepos = jsonPath.find(".json");
		string inferjsonPath = jsonPath.replace(replacepos, 5, "_infer.json");
    	os.open(inferjsonPath, ios::binary);
		os << writer.write(root);
		os.close();
		fp.close();

		return;
	}
	//m_trt->createENG();
	int batchsize = m_trt->param.BatchSize;
	m_trt->inference_init(batchsize, m_trt->param.outputSize);

	vector<cv::Mat> testVal;
	map<string, cv::Mat> dataProb;
	vector<string> imgs;
	cv::Mat img;
	string pattern = m_trt->param.imgDir+ "*."+m_trt->param.imgType;
	vector<cv::String> images_names;
	cv::glob(pattern, images_names, false);
	if(images_names.empty())
	{
		std::cout << "No img files " << std::endl;
		return;
	}
	int i = 0;
	cv::Scalar Mean = cv::Scalar(m_trt->param.mean[0], m_trt->param.mean[1], m_trt->param.mean[2]);
	cv::Scalar Std = cv::Scalar(m_trt->param.std[0], m_trt->param.std[1], m_trt->param.std[2]);
	cv::Size size = { m_trt->param.input_h,m_trt->param.input_w };
	cout << size << "batch size= " << batchsize << endl;
	int flag = 0;
	if (m_trt->param.input_c == 3)
	{
		flag = 1;
	}
	for (auto image_name : images_names)
	{
		if (i < batchsize)
		{
			i++;
			cv::Mat Img = cv::imread(image_name, flag);
			//resize(Img, Img, size, 0, 0, cv::INTER_LINEAR);
			if (flag == 1)
			{
				cv::Mat img;
				Img.convertTo(img, CV_32FC3, 1 / 255.0);
				testVal.push_back(img);
			}
			else {
				testVal.push_back(Img);
			}
			cout << image_name << endl;
			imgs.push_back(image_name);
		}
	}

	std::cout << "out dim : " << m_trt->param.outputSize << std::endl;
	float *data = new float[batchsize*m_trt->param.input_c*m_trt->param.input_h*m_trt->param.input_w];
	int *output = new int[batchsize*m_trt->param.outputSize];

	cv::Mat Transed_t = BlobFromImages(testVal, cv::Size{ m_trt->param.input_w,m_trt->param.input_h }, Mean, Std, true, false);
	//cout<<Transed_t.size<<endl;
	//cv::Mat Transed_t = cv::dnn::blobFromImages(testVal,1.0,cv::Size{m_trt->param.input_h,m_trt->param.input_w},cv::Scalar{0});
	memcpy(data, Transed_t.data, batchsize*m_trt->param.input_c*m_trt->param.input_h*m_trt->param.input_w * sizeof(float));

	
	//m_trt->doInference(data, batchsize, output); // float
	std::cout<<"param.outputSize = " << m_trt->param.outputSize << std::endl;
	m_trt->doInference_int(data, batchsize, output); // float

	////post
	cv::Mat outimg(m_trt->param.input_h, m_trt->param.input_w, CV_8UC1);
	for (int row = 0; row < m_trt->param.input_h; ++row) {
	    uchar* uc_pixel = outimg.data + row * outimg.step;
	    for (int col = 0; col < m_trt->param.input_w; ++col) {
	        uc_pixel[col] = (uchar)output[row*m_trt->param.input_w + col];
	    }
	}
	cv::Mat im_color;
	cv::cvtColor(outimg, im_color, cv::COLOR_GRAY2RGB);
	cv::Mat lut = createLTU(19); // numclass
	cv::LUT(im_color, lut, im_color);
	// false color
	cv::cvtColor(im_color, im_color, cv::COLOR_RGB2GRAY);
	cv::applyColorMap(im_color, im_color, cv::COLORMAP_HOT);
	cv::imshow("False Color Map", im_color);
	//fusion
	//cv::Mat fusionImg;
	//cv::addWeighted(img, 1, im_color, 0.5, 1, fusionImg);
	//cv::imshow("Fusion Img", fusionImg);
	cv::waitKey(0);


}

int main(int argc ,char** argv)
{
    //string JsonPath = argv[1];
    //trt *m_trt = new trt(JsonPath);
    //m_trt->createENG();
    //int batchsize = 1;
    //m_trt->inference_init(batchsize);
    //float *input = new float[batchsize * m_trt->param.input_c * m_trt->param.input_h * m_trt->param.input_w];
    //float *output = new float[batchsize * m_trt->param.outputSize];
    //m_trt->doInference(input,1,output);
	if(argc < 2)
	{
		std::cout << "Parameter error !" << std::endl;
		return -1;
	}
	HRnet(argv[1]);
	
	return 0;
}
