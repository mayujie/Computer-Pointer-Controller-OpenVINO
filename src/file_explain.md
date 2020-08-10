## requirements.txt
The requirements file consists of a list of some of the packages and frameworks that you might need to complete your project. This is not a complete list and you might need more depending on how you solve the project.

## src folder
The source folder contains some code files that will help you get started with your project. 

## input_feeder.py: 
contains an input feeder class that you can use to get input from either a video file or from a webcam. The class has three methods. A load data method that initializes an opencv video captured object with either a video file or the webcam. Next, we have the next_batch function which is a generator that returns successive frames from either the video file or the webcam feed. Finally, the close method closes the video file or the webcam. 
At the top of this file, you will find an example of how you can incorporate this file into your project. So first we will initialize an object of input feeder with the input_type. In case you’re using a video file, you will also need to provide the input_file. After that, you need to call the load_data method to initialize our video captured object. Finally, you can use the next_batch function in a loop. Each batch it returns will a single image. However, you can edit the code here to make it return multiple images. 

## model.py: 
This file contains a skeleton class with methods that will help you load your model, pre-process the inputs of the model and the outputs from the model and also contains a method to run inference on your model. Since each model has different requirements for its inputs and outputs, you will need to create full copies of this file for each model and then finish the to-dos in this file. 

## mouse_controller.py: 
This file contains a class that uses the pyautogui package to help you move the mouse. In the init method, you can set the precision and the speed of the mouse movement. The higher the precision, the more minute the movement will be and the faster the speed. The faster the mouse motion will happen. You can play around with these values to see what gives you the best results. Calling the move method with the x and y output of the gaze estimation model will move your mouse pointer based on your speed and precision settings. 

## bin folder 
contains the video file that you can use if you do not have access to a webcam.
