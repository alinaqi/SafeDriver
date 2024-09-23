# Driver Drowsiness Detection

This project implements a real-time driver drowsiness detection system using computer vision and deep learning techniques. It uses a combination of face detection, eye detection, and a custom-trained CNN model to classify the driver's state as Active, Drowsy, or Sleepy.

## Features

- Real-time video capture and processing
- Face and eye detection using Haar Cascades
- Drowsiness detection using a custom-trained CNN model
- Support for both CPU and CUDA-enabled GPU processing

## Prerequisites

- Python 3.7+
- OpenCV
- PyTorch
- TensorFlow
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/alinaqi/SafeDriver.git
   cd driver-drowsiness-detection
   ```

2. Install the required packages:
   ```
   pip install opencv-python torch torchvision tensorflow numpy
   ```

3. Download the "Driver Drowsiness Dataset (DDD)" from Kaggle (https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd?resource=download) and extract it to a directory named `ddd` in the project root.

## Usage

1. Ensure your webcam is connected and functioning.

2. Run the main script:
   ```
   python drowsiness_detection.py
   ```

3. The script will first train the drowsiness detection model on the DDD dataset. This may take some time depending on your hardware.

4. After training, the webcam feed will open, showing real-time drowsiness detection.

5. Press 'q' to quit the application.

## Project Structure

- `drowsiness_detection.py`: Main script containing all the functions for model training, video capture, and drowsiness detection.
- `ddd/`: Directory containing the Driver Drowsiness Dataset (not included in the repository).

## Customization

- Adjust the `IMG_SIZE`, `BATCH_SIZE`, and `EPOCHS` constants in the script to modify the model training parameters.
- Modify the `create_model()` function to experiment with different CNN architectures.

## Future Improvements

- Implement model saving and loading to avoid retraining on each run.
- Add more sophisticated data augmentation techniques.
- Explore more advanced deep learning architectures for improved accuracy.
- Implement a user interface for easier interaction with the system.

## Contributing

Contributions to improve the project are welcome. Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).