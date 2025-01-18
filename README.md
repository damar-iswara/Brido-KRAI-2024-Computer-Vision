# Brido-KRAI-2024-Computer-Vision
Welcome to the Brido KRAI 2025 Computer Vision repository! This is the code of the program that i make for Brido, Airlangga University Abu Robocon's team use on our very first time qualified to compete at national competition last year. There are only 3 main part on the process in making this whole program. Because the program was created by only one person with minimal experience and knowledge in the realm of computer vision and a small time limit, please forgive me if the program created is less complex and interesting. But for those of you who might be looking for a quick and easy alternative to computer vision, maybe this repo will help.

## Training Model
Abu Robocon's rule stated that R2 need a system that could detect a ball and silo, so i used an object detection model as that was the most effective system for the R2. So, i used YOLOv8 to train model that can detect ball and silo.

**1.Dataset**: We get the dataset manually using our  webcam, so the actual performance of the robot and the model is similar.
**2.Annotation**: For the annotation, we used Roboflow as a platform to annotate, preprocessing, and augmentation of the dataset.
**3.Format**: To adjust the dataset after process above, i also used Roboflow to convert it to the YOLOv8 format.

Roboflow: https://roboflow.com/

## Image Processing (CV2)


## Data Communication
