#include "utils.h"


void mblobFromImages(cv::InputArrayOfArrays images_, cv::OutputArray blob_,
    cv::Size size, const cv::Scalar& mean_, const cv::Scalar& std_, bool swapRB, bool crop)
{
    //CV_TRACE_FUNCTION();
    std::vector<cv::Mat> images;
    images_.getMatVector(images);
    CV_Assert(!images.empty());
    for (int i = 0; i < images.size(); i++)
    {
        cv::Size imgSize = images[i].size();
        if (size == cv::Size())
            size = imgSize;
        if (size != imgSize)
        {
            if (crop)
            {
                float resizeFactor = std::max(size.width / (float)imgSize.width,
                    size.height / (float)imgSize.height);
                resize(images[i], images[i], cv::Size(), resizeFactor, resizeFactor, cv::INTER_LINEAR);
                cv::Rect crop(cv::Point(0.5 * (images[i].cols - size.width),
                    0.5 * (images[i].rows - size.height)),
                    size);
                images[i] = images[i](crop);
            }
            else
                resize(images[i], images[i], size, 0, 0, cv::INTER_LINEAR);
        }
        if (images[i].depth() == CV_8U)
            images[i].convertTo(images[i], CV_32F);
        cv::Scalar mean = mean_;
        cv::Scalar std_num = std_;
        if (swapRB)
        {
            std::swap(mean[0], mean[2]);
            std::swap(std_num[0], std_num[2]);
        }

        images[i] -= mean;
        cv::divide(images[i], std_num, images[i]);
    }

    size_t i, nimages = images.size();
    cv::Mat image0 = images[0];
    int nch = image0.channels();
    CV_Assert(image0.dims == 2);
    cv::Mat image;
    if (nch == 3 || nch == 4)
    {
        int sz[] = { (int)nimages, nch, image0.rows, image0.cols };
        blob_.create(4, sz, CV_32F);
        cv::Mat blob = blob_.getMat();
        cv::Mat ch[4];

        for (i = 0; i < nimages; i++)
        {
            image = images[i];
            CV_Assert(image.depth() == CV_32F);
            nch = image.channels();
            CV_Assert(image.dims == 2 && (nch == 3 || nch == 4));
            CV_Assert(image.size() == image0.size());

            for (int j = 0; j < nch; j++)
                ch[j] = cv::Mat(image.rows, image.cols, CV_32F, blob.ptr((int)i, j));
            if (swapRB)
                std::swap(ch[0], ch[2]);
            split(image, ch);
        }
    }
    else
    {
        CV_Assert(nch == 1);
        int sz[] = { (int)nimages, 1, image0.rows, image0.cols };
        blob_.create(4, sz, CV_32F);
        cv::Mat blob = blob_.getMat();

        for (i = 0; i < nimages; i++)
        {
            cv::Mat image = images[i];
            CV_Assert(image.depth() == CV_32F);
            nch = image.channels();
            CV_Assert(image.dims == 2 && (nch == 1));
            CV_Assert(image.size() == image0.size());

            image.copyTo(cv::Mat(image.rows, image.cols, CV_32F, blob.ptr((int)i, 0)));
        }
    }
}
cv::Mat BlobFromImages(cv::InputArrayOfArrays images, cv::Size size,
    const cv::Scalar& mean, const cv::Scalar& std_num, bool swapRB, bool crop)
{
    //CV_TRACE_FUNCTION();
    cv::Mat blob;
    mblobFromImages(images, blob, size, mean, std_num, swapRB, crop);
    return blob;
}
