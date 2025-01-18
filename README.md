# Brido-KRAI-2024-Computer-Vision
Welcome to the Brido KRAI 2025 Computer Vision repository! This is the code of the program that i make for Brido, Airlangga University Abu Robocon's team use on our very first time qualified to compete at national competition last year. There are only 3 main part on the process in making this whole program. Because the program was created by only one person with minimal experience and knowledge in the field of computer vision and a small time limit, please forgive me if the program created is not complex and interesting. But for those of you who might be looking for a quick and easy alternative to computer vision, maybe this repo will help.

### Full version of the code is at master branch

## Training Model
Abu Robocon's rule stated that R2 need a system that could detect a ball and silo, so i used an object detection model as that was the most effective system for the R2. So, i used YOLOv8 to train model that can detect ball and silo.

*Preprocessing Dataset*
- **Dataset**: We get the dataset manually using our  webcam, so the actual performance of the robot and the model is similar.
- **Annotation**: For the annotation, we used Roboflow as a platform to annotate, preprocessing, and augmentation of the dataset.
- **Format**: To adjust the dataset after process above, i also used Roboflow to convert it to the YOLOv8 format.

Roboflow: https://roboflow.com/


After I got the dataset that match with the YOLOv8 format, next i just need to train the dataset using ultralytics library. To train the model, you just need to do these simple and few line of code.

**Model Training Code**
```python
from ultralytics import YOLO

# Load a pre-trained YOLO model (you can choose n, s, m, l, or x versions)
model = YOLO("yolo9s.pt")

# Start training on your custom dataset
model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640)
```

After that, you just need to wait until the training process ended. After the training process ended, next you can use it straight away or you can change the format of the model depends on the specification of your device. As for me, i convert the model to OpenVino and use quantization to compress the model to fp16.

You can access more information about ultralytics library through their documentation below.
- https://docs.ultralytics.com/#how-can-i-train-a-custom-yolo-model-on-my-dataset
- https://docs.ultralytics.com/modes/export/#arguments
- https://docs.ultralytics.com/integrations/openvino/#reproduce-our-results

## Image Processing (CV2)
Next step is implementing the model using OpenCV/CV2 to do the image processing. Using CV2 library i make a couple of main mechanism for the image processing such as:

**Bounding Box Logic**
```python
    def plot_silo_bboxes(self, results, frame):
    
        closeBid = []
        length = 640
        id = "Silo-0"

        for result in results:
            
            # Extracting the bbox and class name of objects
            boxes = result.boxes.cpu().numpy()
            class_ids = boxes.cls
            xyxys = boxes.xyxy

            # Loop through the coordinate each object and iterate i at the same time
            for i, (x, y, x1, y1) in enumerate(xyxys):

                # Extracting class id to string
                coordinate = [x, y, x1, y1]
                centerBox, lengths = self.center_bbox(coordinate, frame)
                id_name = result.names[int(class_ids[i])]

                print(length)
                print(lengths)
                
                # Filtering the object, silo-1 takes priority
                if id_name == "Silo-1":
                    if abs(length) > abs(lengths):
                        id = id_name
                        length = lengths
                        closeBid = [i for i in coordinate]
                        class_id = id_name                        
                        # print(f"class id: {class_ids[i]}")
                elif id != "Silo-1" and id_name != "Silo-2":
                    if abs(length) > abs(lengths):
                        id = id_name
                        length = lengths
                        class_id = id_name
                        closeBid = [i for i in coordinate]
                        
        if length != 90 and len(closeBid) != 0:
            print(closeBid)
            cv2.rectangle(frame, (int(closeBid[0]), int(closeBid[1])), (int(closeBid[2]), int(closeBid[3])), (50, 205, 50), 2)
            cv2.putText(frame, str(class_id), (int(closeBid[0]), int(closeBid[3]+15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 57), 2, cv2.LINE_AA, False)


        return frame, length
```
The plot_silo_bboxes function identifies and prioritizes bounding boxes around objects detected in a frame, focusing specifically on silos. It iterates through detection results to extract bounding box coordinates, class IDs, and object names. For each object, it calculates the bounding box's center and distance. Priority is given to "Silo-1," with "Silo-2" considered only if no "Silo-1" is detected. The closest bounding box is determined by minimizing the distance, and its coordinates and class ID are stored. If a valid bounding box is identified, a green rectangle is drawn around it in the frame, with its class name displayed as text. This logic helps highlight the most relevant object based on proximity and priority.

**Error Calculation**
```python
    def center_bbox(self, coordinate, frame):
        # Extract the coordinate of a valid object
        x, y, x1, y1 = coordinate

        cx = int((x + x1)/2)   # Center point of X coordinates
        cy  = int((y + y1)/2)   # Center point of Y coordinates

        center = [cx, cy]
        centerFrame = [320, 480]

        # Call a function to calculate the length of a valid object
        errorParam = self.object_length(center, centerFrame)
        error = errorParam[0]
        
        # Create the bbox and line to the valid object
        cv2.circle(frame, center, 3, (0, 255, 0), -1)
        cv2.line(frame, (cx, cy), (320, cy), (0, 0, 0), 2)
        cv2.putText(frame, str(round(errorParam[0], 2)), (int(x+10), int(y+20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1/2, (37, 95, 255), 2, cv2.LINE_AA, False)
        
        return frame, error
```
The center_bbox function calculates the center point of a bounding box for a detected object and determines its horizontal error relative to a reference point in the frame. It takes the bounding box coordinates, calculates the center point, and compares it to a predefined reference center ([320, 480]) using the object_length method. The calculated error is displayed near the bounding box, and a visual indicator (a green circle at the center and a black horizontal line) is added to the frame for better visualization. This function is essential for determining object alignment and highlighting relevant metrics.

**Color Masking**
```python
    def color_masking(self, frame):
        # Blue Color
        # light_color = np.array([90, 100, 100])
        # dark_color = np.array([115, 255, 255])

        # #Red Color
        # light_color = np.array([165, 180, 180])
        # dark_color = np.array([180, 255, 255])

        #Red Color (testing)
        light_color = np.array([0, 25, 75])
        dark_color = np.array([8, 255, 255])

        # Map HSV values to RGB light to dark color range
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, light_color, dark_color)
        b, g, r = [0, 255, 0]

        # Masking the video stream with the color mask
        color_result = cv2.bitwise_and(frame, frame, mask=color_mask)

        # Return the mask result and the bgr value for bbox
        return color_result, b, g, r
```
The color_masking function isolates specific colors within a video frame using a color range in the HSV color space. It defines a range for the target color (adjustable via light_color and dark_color), converts the frame from BGR to HSV, and creates a binary mask where only pixels within the target color range are highlighted. This mask is applied to the frame using bitwise operations to produce the color-isolated output. Additionally, the function specifies the BGR color ([0, 255, 0]) for bounding boxes, which can be used for further annotations. It returns the masked frame and the selected BGR color values.


As for the complete code and the algorithm, just see the file attach on this repo. Those are only the highlight of the main mechanism of the object computer vision system I implemented on the R2.

## Data Communication
**Serial Communication**
```python
            # Integrating bounding boxes and color detection to the frame
            ballFrame, errBall = self.plot_ball_bboxes(predBallFrame, frame)
            siloFrame, errSilo = self.plot_silo_bboxes(predSiloFrame, frame2)
            color_frame, b, g, r = self.color_masking(frame)
            
            print(f"Error Bola: {errBall}")
            print(f"Error Silo: {errSilo}")
            if(a % 2 == 0):
                serialInst.write('S'.encode())
                serialInst.write(f'{errBall}'.encode())
            else:
                serialInst.write('S'.encode())
                serialInst.write(f'{errSilo}'.encode())     
            try:
                response = serialInst.readline().decode('utf-8', errors='ignore').strip() # Read the response from Arduino
                print(f"Response Received: {response}")
            except UnicodeDecodeError as e:
                print("UnicodeDecodeError:", e)
```

This code facilitates communication between a Python script and an Arduino using a serial connection. The Python script processes video frames to detect objects like a ball and a silo, calculates error values (errBall and errSilo), and optionally performs color detection. It sends these error values to the Arduino in a loop, alternating between ball and silo data based on the value of a % 2. Each transmission starts with a command identifier ('S') followed by the error data.

After sending data, the script waits for a response from the Arduino, which it reads and decodes. If decoding fails due to invalid characters, the error is logged. This setup is common in robotics and IoT, where high-level systems process data and send commands to low-level controllers that execute tasks and provide feedback. Here, the Arduino likely uses the error values to control actuators or make adjustments in a vision-guided system.
