# Computer Pointer Controller
_______________
## Introduction
Computer Pointer Controller is an application which use a gaze detection model to control the mouse pointer of you computer.
The position of mouse pointer will change by following the user's Gaze. The [Gaze Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) model is used to estimate the gaze of the user's eyes and then feed the result into `pyautogui` module to change the position of mouse pointer. 

The pipline of application as shown below:
![pipline](https://github.com/mayujie/Computer-Pointer-Controller-OpenVINO/blob/master/bin/pipeline.png?raw=true)

### LiveDemo:
Recorded video of running the project: [Demo video of project](https://youtu.be/ZJ8y--zcBag)

### Screenshot:
![show_app](https://github.com/mayujie/Computer-Pointer-Controller-OpenVINO/blob/master/bin/show_app.PNG?raw=true)

## Project Set Up and Installation
_______________
**1.Prerequisites** 
- [Install Intel® Distribution of OpenVINO™ toolkit for Windows* 10](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html#model_optimizer_configuration_steps) or you can choose install in Linux system.
- The `requirments.txt` in project directory needs to be installed. Using command: `pip3 install -r requirements.txt`

**2.Environment setup**
Initialize openVINO environment (command in cmd)
**Important!!!**
```sh
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\ && setupvars.bat
```
**3.Download the required model**
- Download the required models:
    - [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
    - [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
    - [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
    - [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

- These can be downloaded using the `model downloader`. 
- cd to project directory and follow command below to download the models.
 ```sh
    python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name face-detection-adas-binary-0001
    
    python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name head-pose-estimation-adas-0001
    
    python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name landmarks-regression-retail-0009
    
    python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name gaze-estimation-adas-0002
```


The source structure of project as showm below: 
```
E:\Intel-AI\Computer-Pointer-Controller-OpenVINO>tree /a /f
Folder PATH listing for volume entertainment
Volume serial number is 0003-B93B
E:.
|   .Instructions.md.swp
|   README.md
|   requirements.txt
|
+---bin
|       .gitkeep
|       demo.mp4
|       pipeline.png
|       show_app.PNG
|
+---intel
|   +---face-detection-adas-binary-0001
|   |   \---FP32-INT1
|   |           face-detection-adas-binary-0001.bin
|   |           face-detection-adas-binary-0001.xml
|   |
|   +---gaze-estimation-adas-0002
|   |   +---FP16
|   |   |       gaze-estimation-adas-0002.bin
|   |   |       gaze-estimation-adas-0002.xml
|   |   |
|   |   +---FP16-INT8
|   |   |       gaze-estimation-adas-0002.bin
|   |   |       gaze-estimation-adas-0002.xml
|   |   |
|   |   \---FP32
|   |           gaze-estimation-adas-0002.bin
|   |           gaze-estimation-adas-0002.xml
|   |
|   +---head-pose-estimation-adas-0001
|   |   +---FP16
|   |   |       head-pose-estimation-adas-0001.bin
|   |   |       head-pose-estimation-adas-0001.xml
|   |   |
|   |   +---FP16-INT8
|   |   |       head-pose-estimation-adas-0001.bin
|   |   |       head-pose-estimation-adas-0001.xml
|   |   |
|   |   \---FP32
|   |           head-pose-estimation-adas-0001.bin
|   |           head-pose-estimation-adas-0001.xml
|   |
|   \---landmarks-regression-retail-0009
|       +---FP16
|       |       landmarks-regression-retail-0009.bin
|       |       landmarks-regression-retail-0009.xml
|       |
|       +---FP16-INT8
|       |       landmarks-regression-retail-0009.bin
|       |       landmarks-regression-retail-0009.xml
|       |
|       \---FP32
|               landmarks-regression-retail-0009.bin
|               landmarks-regression-retail-0009.xml
|
\---src
    |   face_detection.py
    |   facial_landmarks_detection.py
    |   file_explain.md
    |   gaze_estimation.py
    |   head_pose_estimation.py
    |   input_feeder.py
    |   main.py
    |   model.py
    |   mouse_controller.py
    |   Project_log.log
    |
    \---__pycache__
            face_detection.cpython-37.pyc
            facial_landmarks_detection.cpython-37.pyc
            gaze_estimation.cpython-37.pyc
            head_pose_estimation.cpython-37.pyc
            input_feeder.cpython-37.pyc
            mouse_controller.cpython-37.pyc

```

## Demo
_______________
**1. cd to `src` folder first**

**2. Template of run the `main.py`**
```
python main.py -f <Path of xml file for face detection model> -fl <Path of xml file for facial landmarks detection model> -hp <Path of xml file for head pose estimation model> -g <Path of xml file for gaze estimation model> -i <Path of input video file or enter cam for feeding input from webcam> -d <choose the device to run the model (default CPU)> -flags <select the visualization: fd fld hp ge>
```

**3. Examples of run `the main.py`:**
- Running model with percision **FP32** in **CPU**:
```sh
python main.py -f "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml" -fl "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml" -hp "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml" -g "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002.xml" -i "\Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4" -d CPU -flags fd fld hp ge
```

- Running model with percision **FP16** in **CPU**:
```sh
python main.py -f "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml" -fl "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009.xml" -hp "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml" -g "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002.xml" -i "\Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4" -d CPU -flags fd fld hp ge
```

- Running model with percision **FP16-INT8** in **CPU**:
```sh
python main.py -f "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml" -fl "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP16-INT8\landmarks-regression-retail-0009.xml" -hp "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP16-INT8\head-pose-estimation-adas-0001.xml" -g "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP16-INT8\gaze-estimation-adas-0002.xml" -i "\Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4" -d CPU -flags fd fld hp ge
```

- Running model with percision **FP32** in **CPU** and input through **webcam**
```sh
python main.py -f "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml" -fl "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml" -hp "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml" -g "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002.xml" -i cam -d CPU -flags fd fld hp ge
```

## Documentation
_______________

### Command line agruments 
**Try command `python main.py -h` to get help for command line arguments of the application** 
```
E:\Intel-AI\Computer-Pointer-Controller-OpenVINO\src>python main.py -h
usage: main.py [-h] -f FACEDETECTIONMODEL -fl FACIALLANDMARKMODEL -hp
               HEADPOSEMODEL -g GAZEESTIMATIONMODEL -i INPUT
               [-flags PREVIEWFLAGS [PREVIEWFLAGS ...]] [-l CPU_EXTENSION]
               [-prob PROB_THRESHOLD] [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -f FACEDETECTIONMODEL, --facedetectionmodel FACEDETECTIONMODEL
                        Specify Path to .xml file of Face Detection model.
  -fl FACIALLANDMARKMODEL, --faciallandmarkmodel FACIALLANDMARKMODEL
                        Specify Path to .xml file of Facial Landmark Detection
                        model.
  -hp HEADPOSEMODEL, --headposemodel HEADPOSEMODEL
                        Specify Path to .xml file of Head Pose Estimation
                        model.
  -g GAZEESTIMATIONMODEL, --gazeestimationmodel GAZEESTIMATIONMODEL
                        Specify Path to .xml file of Gaze Estimation model.
  -i INPUT, --input INPUT
                        Specify Path to video file or enter cam for webcam
  -flags PREVIEWFLAGS [PREVIEWFLAGS ...], --previewFlags PREVIEWFLAGS [PREVIEWFLAGS ...]
                        Specify the flags from fd, fld, hp, ge like --flags fd
                        hp fld (Seperate each flag by space)for see the
                        visualization of different model outputs of each
                        frame,fd for Face Detection, fld for Facial Landmark
                        Detectionhp for Head Pose Estimation, ge for Gaze
                        Estimation.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with thekernels impl.
  -prob PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for model to detect the face
                        accurately from the video frame.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
```
## Benchmarks
_______________

- FP32, FP16, FP16-INT8 tested on my CPU: Intel(R) Core(TM)i7-3632QM CPU @ 2.20GHz
- Checked total load model, total inference time, FPS, model size

#### FP32 Project_log 
```
INFO:root:Model Load time: 0.9885485172271729
INFO:root:Inference time: 28.26241898536682
INFO:root:FPS: 2.0877565463552723
ERROR:root:VideoStream ended...
```

#### FP16 Project_log 
```
INFO:root:Model Load time: 0.9749979972839355
INFO:root:Inference time: 28.06306004524231
INFO:root:FPS: 2.1026372059871705
ERROR:root:VideoStream ended...
```
#### FP16-INT8 Project_log 
```
INFO:root:Model Load time: 1.3344995975494385
INFO:root:Inference time: 27.815059900283813
INFO:root:FPS: 2.12077641984184
ERROR:root:VideoStream ended...
```

### FP32
|Model|Type|Size|
| ------ | ------ | ------ |
|face-detection-adas-binary-0001|FP32-INT1|1.86 MB|
|head-pose-estimation-adas-0001	|FP32|7.34 MB|
|landmarks-regression-retail-0009|FP32|786 KB|
|gaze-estimation-adas-0002|FP32|7.24 MB|

### FP16
|Model|Type|Size|
| ------ | ------ | ------ |
|face-detection-adas-binary-0001|FP32-INT1|1.86 MB|
|head-pose-estimation-adas-0001	|FP16|3.69 MB|
|landmarks-regression-retail-0009|FP16|413 KB|
|gaze-estimation-adas-0002|FP16|3.69 MB|

### FP16-INT8
|Model|Type|Size|
| ------ | ------ | ------ |
|face-detection-adas-binary-0001|FP32-INT1|1.86 MB|
|head-pose-estimation-adas-0001	|FP16-INT8|2.05 MB|
|landmarks-regression-retail-0009|FP16-INT8|314 KB|
|gaze-estimation-adas-0002|FP16-INT8|2.09 MB|

### Comparison
||Total Model Load time (sec)|Total Inference Time (sec)|FPS|
| ------ | ------ | ------ | ------ |
|FP32|0.9885485172271729|28.26241898536682|2.0877565463552723|
|FP16|0.9749979972839355|28.06306004524231|2.1026372059871705|
|FP16-INT8|1.3344995975494385|27.815059900283813|2.12077641984184|

## Results
_______________
- For different precision, the model size decreases follow the order of FP32 > FP16 > FP16-INT8. The inference time follows the same order in this case, the INT8 is faster than FP16 and FP32 is slower than FP32. Lower precision model uses less memory. However, remember lower precision of model also lose the accuracy of model.
- Memory Access of FP16 is half the size compared with FP32, which reduces memory usage of a neural network. FP16 data transfers are faster than FP32, which improves speed (TFLOPS) and performance.
- The Model load time and FPS of them are almost the same, besides model only load only when model initializes.
- In order to achieving the most reasonable combination, we do not want too longer inference time also too low accuracy. Moreover, in some specific scenario such as low budget. We do not want to waste storage attempt to get very high accuracy. To achieve the balance, we need to consider the volume of sacrifice.



## Stand Out Suggestions
_______________
- Use the VTune Amplifier to find hotspots in inference engine pipline.
- Build an inference pipeline for both video file and webcam feed as input. Allow the user to select their input option in the command line arguments. 
- Benchmark the running times of different parts of the preprocessing and inference pipeline and let the user specify a CLI argument if they want to see the benchmark timing. Use the get_perf_counts API to print the time it takes for each layer in the model. 

### Edge Cases
_______________
- If there will multiple face situation, the model only extracts one face to control the mouse pointer and ignore the other faces.
- If due to certain reason, model couldn't detect the face. It will continue process another frame untill it detects face or keyboard interrupt to exit the program.

