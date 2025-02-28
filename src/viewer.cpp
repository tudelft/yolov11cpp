#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;

int main(int argc, char const *argv[])
{
    const std::string videoSource = "./input.mov"; // your usb cam device


    cv::VideoCapture cap;

    // configure the best camera to iphone 11
    cap.open(videoSource, cv::CAP_FFMPEG);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open the camera!\n";
        return -1;
    }

    for(;;)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Error: Could not read a frame!\n";
            break;
        }

        // Display the frame
        cv::imshow("input", frame);

        if (cv::waitKey(1) >= 0)
        {
            break;
        }

    }


    return 0;
}
