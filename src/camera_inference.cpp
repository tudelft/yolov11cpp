#include <iostream>
#include <vector>
#include <thread>
#include <atomic>

#include <opencv2/highgui/highgui.hpp>

#include "./ia/YOLO11.hpp" 

// Include the bounded queue
#include "tools/BoundedThreadSafeQueue.hpp"


int main()
{
    // Configuration parameters
    const bool isGPU = false;
    const std::string labelsPath = "./classes.txt";
    const std::string modelPath = "./best.onnx";
    const std::string videoSource = "./input.mov"; // your usb cam device
    const std::string outputPath = "./output.mp4"; // path for output video file
    
    // Set rotation angle based on input video orientation (0, 90, 180, or 270 degrees)

    // Initialize YOLO detector
    // YOLO9Detector detector(modelPath, labelsPath, isGPU);
    YOLO11Detector detector(modelPath, labelsPath, isGPU);

    // Open video capture
    cv::VideoCapture cap;

    // configure the best camera to iphone 11
    cap.open(videoSource, cv::CAP_FFMPEG);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open the camera!\n";
        return -1;
    }
    
    // Get video properties for the writer
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    // Initialize video writer
    cv::VideoWriter videoWriter;
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // MP4 codec
    
    // Open the video writer
    bool isWriterOpened = videoWriter.open(outputPath, fourcc, fps, cv::Size(width, height), true);
    if (!isWriterOpened) {
        std::cerr << "Error: Could not open video writer!\n";
        return -1;
    }
    
    std::cout << "Recording output to: " << outputPath << std::endl;
    std::cout << "Press 'q' to stop recording and exit" << std::endl;

    for (;;)
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

        // Perform detection on the rotated frame
        std::vector<Detection> detections = detector.detect(frame);

        // Create a copy for output with detections drawn
        cv::Mat outputFrame = frame.clone();
        
        // Draw bounding boxes and masks on the frame
        detector.drawBoundingBoxMask(outputFrame, detections);

        // Write the processed frame to the output video
        videoWriter.write(outputFrame);
        
        // Display the frame
        cv::imshow("Detections", outputFrame);

        // Use a small delay and check for 'q' key press to quit
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }
    
    // Release resources
    cap.release();
    videoWriter.release();
    cv::destroyAllWindows();
    
    std::cout << "Video processing completed. Output saved to: " << outputPath << std::endl;

    return 0;
}
