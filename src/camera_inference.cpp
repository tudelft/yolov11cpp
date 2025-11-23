#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <filesystem>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "./ia/YOLO11.hpp" 


namespace fs = std::filesystem;


int main()
{

    // Configuration parameters
    const bool isGPU = false;

    const std::string logsPath = "/home/mavlab/Desktop/Logs/624/camera";

    const std::string labelsPath = "../../drone_labelling/models/classes.txt";
    // const std::string modelPath = "./best_optmized.onnx";
    const std::string modelPath = "../../drone_labelling/models/DroNet-v1_2_0-A2RL.onnx";
    const std::string videoSource = "./input.mov"; // your usb cam device
    const std::string outputPath = "./output.mp4"; // path for output video file
    
    // Use the same default thresholds as Ultralytics CLI
    const float confThreshold = 0.2f;  // Match Ultralytics default confidence threshold
    const float iouThreshold = 0.35f;   // Match Ultralytics default IoU threshold
    
    std::cout << "Initializing YOLOv11 detector with model: " << modelPath << std::endl;
    std::cout << "Using confidence threshold: " << confThreshold << ", IoU threshold: " << iouThreshold << std::endl;
    
    // read model 
    std::cout << "Loading model and labels..." << std::endl;

    // Initialize YOLO detector
    YOLO11Detector detector(modelPath, labelsPath, isGPU);

    // Get a list of images in the logsPath directory
    // Check folder
    if (!fs::exists(logsPath)) {
        std::cerr << "Folder not found: " << logsPath << std::endl;
        return -1;
    } else {
        std::cout << "Found folder: " << logsPath << std::endl;
    }

    
    // Get video properties for the writer
    // double fps = cap.get(cv::CAP_PROP_FPS);
    // int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    // int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
   
    int width = 820;
    int height = 616;
    int fps = 120;

    // Initialize video writer
    cv::VideoWriter videoWriter;
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // Motion JPEG codec
    
    // Open the video writer
    bool isWriterOpened = videoWriter.open(outputPath, fourcc, fps, cv::Size(width, height), true);
    if (!isWriterOpened) {
        std::cerr << "Error: Could not open video writer!\n";
        return -1;
    }
    
    std::cout << "Recording output to: " << outputPath << std::endl;
    std::cout << "Press 'q' to stop recording and exit" << std::endl;

    int frame_count = 0;
    double total_time = 0.0;


    // Create a vector to store the file paths and their last write times
    std::vector<std::pair<std::string, std::filesystem::file_time_type>> files;

    // Iterate through the directory and collect file paths and their last write times
    for (const auto& entry : fs::directory_iterator(logsPath)) {
        if (entry.path().extension() == ".jpeg") {
            files.emplace_back(entry.path().string(), fs::last_write_time(entry));
        }
    }

    // Sort the files by their last write time
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    // Process the sorted files
    for (const auto& file : files) {
        std::cout << "Loading: " << file.first << std::endl;
        cv::Mat frame = cv::imread(file.first);

        if (frame.empty()) {
            std::cerr << "Failed to load image!" << std::endl;
            continue;
        }

        // Resize the image to 640 x 640
        cv::resize(frame, frame, cv::Size(640, 640));

        // Display the frame
        // cv::imshow("input", frame);

        // Perform detection with the updated thresholds
        std::vector<Detection> detections = detector.detect(frame, confThreshold, iouThreshold);

        frame_count++;
        total_time += 8.33;

        // Create a copy for output with detections drawn
        cv::Mat outputFrame = frame.clone();

        // Draw bounding boxes and masks on the frame
        detector.drawBoundingBoxMask(outputFrame, detections);



        // Write the processed frame to the output video
        videoWriter.write(outputFrame);

        // Display the frame
        cv::imshow("Detections", outputFrame);

        // Use a small delay and check for 'q' key press to quit
        if (cv::waitKey(100) == 'q') {
            break;
        }
    }
    
    // Release resources
    // cap.release();
    videoWriter.release();
    cv::destroyAllWindows();
    
    std::cout << "Video processing completed. Output saved to: " << outputPath << std::endl;
    std::cout << "Average FPS: " << (1000.0 / (total_time / frame_count)) << std::endl;

    return 0;
}
