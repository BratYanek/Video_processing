#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // Define the size of the images used for training
    const int imageSize = 20; // Size of the image (e.g., 20x20)
    const int featureSize = imageSize * imageSize; // Number of features

    // Create k-NN model
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();

    std::vector<cv::Mat> trainingImages;
    std::vector<int> labels;

    // Load images and labels
    std::string dataPath = "database/";
    for (const auto& entry : fs::directory_iterator(dataPath)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            cv::Mat image = cv::imread(filePath, cv::IMREAD_GRAYSCALE);

            if (image.empty()) {
                std::cerr << "Error loading image: " << filePath << std::endl;
                continue;
            }

            // Resize to ensure consistency
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(imageSize, imageSize));

            // Flatten the image to a row vector
            cv::Mat flat = resized.reshape(1, 1);

            // Convert to float
            flat.convertTo(flat, CV_32F);

            // Determine the label from the file name
            std::string fileName = entry.path().filename().string();
            char label = fileName[6]; // Extract the letter (e.g., letter_a_001.jpg -> 'a')
            labels.push_back(label - 'a'); // Convert letter to integer label

            trainingImages.push_back(flat);
        }
    }

    if (trainingImages.empty() || labels.empty()) {
        std::cerr << "No training data found." << std::endl;
        return -1;
    }

    // Convert vectors to matrices
    cv::Mat trainingData;
    cv::vconcat(trainingImages, trainingData);

    cv::Mat labelsMat(labels);

    // Train the k-NN model
    knn->setDefaultK(3);
    knn->train(trainingData, cv::ml::ROW_SAMPLE, labelsMat);
    knn->save("knn_model.yml");

    std::cout << "Model trained and saved to knn_model.yml" << std::endl;

    return 0;
}
