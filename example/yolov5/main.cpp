#include "trt.h"
#include "utils.h"
#include "yololayer.h"
#include <io.h>
#include <direct.h>
#include <opencv2/opencv.hpp>

cv::Rect get_rect(cv::Mat& img, float bbox[4], int& INPUT_W, int& INPUT_H) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }
    else {
        l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.conf > b.conf;
}

void nms(std::vector<Yolo::Detection>& res, float* output, int& MAX_OUTPUT_BBOX_COUNT, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
        //cout<<det.conf<<det.class_id<<det.bbox[0]<<det.bbox[1]<<det.bbox[2]<<det.bbox[3];
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}
static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}
void yolo_img(trt* m_trt, int& batchsize)
{
    vector<cv::Mat> testVal;
    vector<string> imgs;
    string pattern = m_trt->param.imgDir + "*." + m_trt->param.imgType;//"D:/qt_project/tensorrtCV/model/yolo/*.jpg";
    vector<cv::String> images_names;
    cv::glob(pattern, images_names, false);

    string outputPath = m_trt->param.imgDir + "output/";
    if (_access(outputPath.c_str(), 0) == -1)
    {
        _mkdir(outputPath.c_str());
    }
    int inputH = m_trt->param.input_h;
    int inputW = m_trt->param.input_w;
    int flag = 1;
    if (m_trt->param.input_c == 1)
        flag = 0;
    float* input = new float[batchsize * m_trt->param.input_c * inputH * inputW];
    float* output = new float[batchsize * m_trt->param.outputSize];
    int inferBatch = 0;
    if (images_names.size() == 0)
    {
        cout << "no " << m_trt->param.imgType << " in " << m_trt->param.imgDir << endl;
        return;
    }
    for (auto image_name : images_names)
    {
        cout << image_name << endl;
        cv::Mat Img = cv::imread(image_name, flag);
        cv::Mat img = preprocess_img(Img, inputW, inputH);
        testVal.push_back(img);
        imgs.push_back(image_name);
        inferBatch++;
        if (testVal.size() != batchsize && image_name != *(images_names.end() - 1))
        {
            continue;
        }
        cout << testVal.size() << "  " << inferBatch << endl;
        cv::Mat Data = cv::dnn::blobFromImages(testVal, 1.0, cv::Size{ inputH,inputW }, cv::Scalar{ 0 }, true);//BGR -> RGB

        memcpy(input, Data.data, inferBatch * m_trt->param.input_c * inputH * inputW * sizeof(float));

        m_trt->doInference(input, inferBatch, output);

        std::vector<std::vector<Yolo::Detection>> batch_res(inferBatch);
        int MAX_OUTPUT_BBOX_COUNT = (m_trt->param.outputSize - 1) / 6;

        float CONF_THRESH = 0.5;
        float NMS_THRESH = 0.5;

        for (int j = 0; j < inferBatch; j++)
        {
            auto& res = batch_res[j];
            nms(res, &output[j * m_trt->param.outputSize], MAX_OUTPUT_BBOX_COUNT, CONF_THRESH, NMS_THRESH);
        }

        for (int b = 0; b < inferBatch; b++) {
            auto& res = batch_res[b];
            //        ofstream openfile("H:/myGitHub/tensorrtF/model/yolov5/test/trt_results.txt");
            //        for(int n=0;n<res.size();n++)
            //        {
            //            for(int i = 0;i < 4;i++)
            //            {
            //                openfile<<res[n].bbox[i]<<endl;
            //            }
            //            openfile<<res[n].conf<<endl;
            //            openfile<<res[n].class_id<<endl;
            //        }
            //        openfile.close();
            cv::Mat img = cv::imread(imgs[b], 1);
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(img, res[j].bbox, inputW, inputH);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0x00, 0x00), 2);
            }
            string outPath = imgs[b].replace(0, m_trt->param.imgDir.size(), outputPath);//"D:/qt_project/tensorrtCV/model/test/1.jpg";
            cv::imwrite(outPath, img);
        }
        imgs.clear();
        testVal.clear();
        batch_res.clear();
        inferBatch = 0;
    }
}

void yolo_vedio(trt* m_trt, int& batchsize)
{
    string vedios = m_trt->param.imgDir + "*" + m_trt->param.imgType;//"D:/qt_project/tensorrtCV/model/yolo/test.mp4";
    vector<cv::String> vedios_names;
    cv::glob(vedios, vedios_names, false);
    string outputPath = m_trt->param.imgDir + "output/";
    if (_access(outputPath.c_str(), 0) == -1)
    {
        _mkdir(outputPath.c_str());
    }
    int inputH = m_trt->param.input_h;
    int inputW = m_trt->param.input_w;
    float* input = new float[batchsize * m_trt->param.input_c * inputH * inputW];
    float* output = new float[batchsize * m_trt->param.outputSize];
    for (auto vedio_name : vedios_names)
    {
        cv::VideoCapture capture(vedio_name);
        if (!capture.isOpened())
        {
            cout << "can not open vedio file, please check it : " << vedio_name << endl;
            return;
        }
        cv::Size size = cv::Size((int)capture.get(cv::CAP_PROP_FRAME_WIDTH), (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = capture.get(cv::CAP_PROP_FPS);
        double allFps = capture.get(cv::CAP_PROP_FRAME_COUNT);
        cout << allFps << endl;
        cv::VideoWriter writer;
        string vedioOputPath = ((string)vedio_name).replace(0, m_trt->param.imgDir.size(), outputPath);
        vedioOputPath = vedioOputPath.substr(0, vedioOputPath.size() - 3);
        vedioOputPath = vedioOputPath + "avi";
        cout << "vedio output Path :" << vedioOputPath << endl;
        writer.open(vedioOputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, size, true);
        if (!writer.isOpened())
        {
            cout << "write vedio error! please check it : " << vedioOputPath << endl;
            return;
        }
        cv::Mat img;
        vector<cv::Mat> testVal;
        vector<cv::Mat> imgs;
        auto start_time = std::chrono::system_clock::now();
        int numFps = 0;
        int inferBatch = 0;
        while (capture.read(img)) {
            imgs.push_back(img);
            cv::Mat Img = preprocess_img(img, inputW, inputH);
            testVal.push_back(Img);
            inferBatch++;
            numFps++;
            if (inferBatch != batchsize && numFps < allFps)
            {
                continue;
            }
            cv::Mat Data = cv::dnn::blobFromImages(testVal, 1.0, cv::Size{ inputH,inputW }, cv::Scalar{ 0 }, true);
            memcpy(input, Data.data, inferBatch * m_trt->param.input_c * inputH * inputW * sizeof(float));

            m_trt->doInference(input, inferBatch, output);

            std::vector<std::vector<Yolo::Detection>> batch_res(inferBatch);
            int MAX_OUTPUT_BBOX_COUNT = (m_trt->param.outputSize - 1) / 6;

            float CONF_THRESH = 0.5;
            float NMS_THRESH = 0.5;

            for (int j = 0; j < inferBatch; j++)
            {
                auto& res = batch_res[j];
                nms(res, &output[j * m_trt->param.outputSize], MAX_OUTPUT_BBOX_COUNT, CONF_THRESH, NMS_THRESH);
            }

            for (int b = 0; b < inferBatch; b++) {
                auto& res = batch_res[b];
                cv::Mat writeImg = imgs[b];
                for (size_t j = 0; j < res.size(); j++) {
                    cv::Rect r = get_rect(writeImg, res[j].bbox, inputW, inputH);
                    cv::rectangle(writeImg, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    cv::putText(writeImg, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0x00, 0x00), 2);
                }
                writer.write(writeImg);
            }

            imgs.clear();
            testVal.clear();
            batch_res.clear();
            inferBatch = 0;

        }
        capture.release();
        writer.release();
        auto end_time = std::chrono::system_clock::now();
        float duration;
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        cout << "all time : " << duration << endl;
    }
}
int main()
{
    string JsonPath = "D:/qt_project/tensorrtF/model/yolo/yolov5s.json";
    trt* m_trt = new trt(JsonPath);
    if (m_trt->param.createENG)
        m_trt->createENG();
    if (!m_trt->param.doInfer)
        return 0;
    int batchsize = m_trt->param.BatchSize;
    int outputSize = m_trt->param.outputSize;
    m_trt->inference_init(batchsize, outputSize);
    string imgType = "jpg png bmp jpeg";
    string::size_type imgIdx = imgType.find(m_trt->param.imgType);
    if (imgIdx != string::npos)
        yolo_img(m_trt, batchsize);
    string vedioType = "mp4 avi";
    string::size_type vedioIdx = vedioType.find(m_trt->param.imgType);
    if (vedioIdx != string::npos)
        yolo_vedio(m_trt, batchsize);
    return 0;
}
