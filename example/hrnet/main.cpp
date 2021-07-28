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

void HRnet(std::string jsonPath)
{
	//string jsonPath = "/mnt/f/LearningCode/hrnet/hrnet_w48.json";
	trt *m_trt = new trt(jsonPath);
	if(m_trt->param.createENG)
	{
		m_trt->createENG();
	}

	int batchsize = m_trt->param.BatchSize;
	m_trt->inference_init(batchsize);
	std::cout << "out dim : " << m_trt->param.outputSize << std::endl;
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
	cout << size << "batch size= " << batchsize << endl;
	int flag = 0;
	if (m_trt->param.input_c == 3)
	{
		flag = 1;
	}
	float *data = new float[batchsize*m_trt->param.input_c*m_trt->param.input_h*m_trt->param.input_w];
	int *output = new int[batchsize*m_trt->param.outputSize];
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
				Img.convertTo(img, CV_32FC3);
				testVal.push_back(img);
			}
			else {
				testVal.push_back(Img);
			}
			cout << image_name << endl;
			imgs.push_back(image_name);
		}
	}

	cv::Mat Transed_t = cv::dnn::blobFromImages(testVal,1.0,cv::Size{m_trt->param.input_h,m_trt->param.input_w},cv::Scalar{0},true);//BGR -> RGB
	memcpy(data, Transed_t.data, batchsize*m_trt->param.input_c*m_trt->param.input_h*m_trt->param.input_w * sizeof(float));
	
	//m_trt->doInference(data, batchsize, output); // float
	m_trt->doInference_int(data, batchsize, output); 

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
	if(argc < 2)
	{
		std::cout << "Parameter error !" << std::endl;
		return -1;
	}
	HRnet(argv[1]);
	
	return 0;
}
