#include <iostream>
#include <string>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

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
  cv::Mat frame, gray, blurred, edges;

  int imageCounter = 0; // To keep track of saved images
  char letter;          // To store the letter label

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
    cv::GaussianBlur(gray, blurred, cv::Size(11, 11), 0);
    cv::Canny(blurred, edges, 10, 10);

    // Display the grayscale frame
    cv::imshow("Camera", edges);

    char key = cv::waitKey(1);

    // Wait for 1 ms and check if 'q' is pressed to exit
    if (key == 'q') {
      break;
    }

    if (key >= 'a' && key <= 'z') {
      letter = key;
      std::string fileName = "database/letter_" + std::string(1, letter) + "_" +
                             std::to_string(imageCounter++) + ".jpg";

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
