#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

int main() {
    // Load the trained k-NN model
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::load("knn_model.yml");

    if (knn->empty()) {
        std::cerr << "Error: Unable to load k-NN model." << std::endl;
        return -1;
    }

    // Open the default camera (camera index 0)
    cv::VideoCapture cap(0);

    // Check if the camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

    // Create a window to display the video
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    // Variable to hold the frames
    cv::Mat frame;
    cv::Mat gray;
    cv::Mat floatImage;
    cv::Mat resized;
    cv::Mat flattened;
    int featureSize = 400; // Example size for 20x20 image (20*20=400 features)

    while (true) {
        // Capture each frame
        cap >> frame;

        // Check if the frame was captured properly
        if (frame.empty()) {
            std::cerr << "Error: Empty frame" << std::endl;
            break;
        }

        // Convert the frame to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Resize the grayscale image to match the input size of the model
        cv::resize(gray, resized, cv::Size(20, 20)); // Assuming 20x20 is the feature size

        // Convert the resized image to float
        resized.convertTo(floatImage, CV_32F);

        // Flatten the image matrix to a single row (1 x N)
        flattened = floatImage.reshape(1, 1);

        // Check that the feature vector has the correct size
        if (flattened.cols != featureSize) {
            std::cerr << "Error: Feature vector size mismatch" << std::endl;
            continue;
        }

        // Perform prediction
        cv::Mat results;
        cv::Mat neighborResponses;
        cv::Mat dists;

        try {
            knn->findNearest(flattened, 3, results, neighborResponses, dists);

            // Display the result on the frame
            int label = static_cast<int>(results.at<float>(0, 0));
            std::string text = "Predicted: " + std::to_string(label);
            cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV Exception: " << e.what() << std::endl;
        }

        // Display the frame
        cv::imshow("Camera", frame);

        // Wait for 1 ms and check if 'q' is pressed to exit
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the camera and close all windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
