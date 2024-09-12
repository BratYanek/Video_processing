#include <filesystem>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>

namespace fs = std::filesystem;

// Function to generate the next unique filename
std::string getNextFileName(const std::string &baseName) {
  int counter = 1;
  std::string fileName = baseName + "_001.jpg";

  while (fs::exists(fileName)) {
    fileName = baseName + "_" +
               std::string(3 - std::to_string(counter).length(), '0') +
               std::to_string(counter++) + ".jpg";
  }

  return fileName;
}

int main() {
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
  cv::Mat frame, gray, blurred, edges, resized;

  char letter; // To store the letter label

  // Desired size for the resized frames
  cv::Size desiredSize(640, 480);

  while (true) {
    // Capture each frame
    cap >> frame;

    // Check if the frame was captured properly
    if (frame.empty()) {
      std::cerr << "Error: Empty frame" << std::endl;
      break;
    }

    // Resize the frame to the desired size
    cv::resize(frame, resized, desiredSize);

    // Convert the frame to grayscale
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(11, 11), 0);
    cv::Canny(blurred, edges, 20, 20);

    // Display the processed frame
    cv::imshow("Camera", edges);

    char key = cv::waitKey(1);

    // Wait for 1 ms and check if 'q' is pressed to exit
    if (key == 'q') {
      break;
    }

    if (key >= 'a' && key <= 'z') {
      letter = key;
      std::string baseFileName = "database/letter_" + std::string(1, letter);
      std::string fileName = getNextFileName(baseFileName);

      // Save the current frame as an image with a label
      cv::imwrite(fileName, edges);

      std::cout << "Saved image: " << fileName << std::endl;
    }
  }

  // Release the camera and close all windows
  cap.release();
  cv::destroyAllWindows();

  return 0;
}
