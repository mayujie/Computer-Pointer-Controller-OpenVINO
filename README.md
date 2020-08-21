# Computer Pointer Controller

## Introduction
Computer Pointer Controller is an application which use a gaze detection model to control the mouse pointer of you computer.
The position of mouse pointer will change by following the user's Gaze. The [Gaze Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) model is used to estimate the gaze of the user's eyes and then feed the result into `pyautogui` module to change the position of mouse pointer. 

The pipline of application as shown below:
![pipline](https://github.com/mayujie/Computer-Pointer-Controller-OpenVINO/blob/master/bin/pipeline.png?raw=true)

### LiveDemo:
Recorded video of running the project: [Demo video of project](https://youtu.be/ArcQ60vY0Z0)

### Screenshot:
![show_app](https://github.com/mayujie/Computer-Pointer-Controller-OpenVINO/blob/master/bin/show_app.PNG?raw=true)

## Project Set Up and Installation
_______________
**1.Prerequisites** 
- [Install Intel® Distribution of OpenVINO™ toolkit for Windows* 10](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html#model_optimizer_configuration_steps) or you can choose install in Linux system.
- The `requirments.txt` in project directory needs to be installed. Using command: 
    - `pip3 install -r requirements.txt`

**2.Environment setup**
Initialize openVINO environment (command in cmd)

**!!!Important!!!**
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
Volume serial number is 00000060 0003:B93B
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
    |   test.py
    |   testarg.py
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
python main.py -fd <Path of xml file for face detection model> -fl <Path of xml file for facial landmarks detection model> -hp <Path of xml file for head pose estimation model> -ge <Path of xml file for gaze estimation model> -i <Path of input video file or enter cam for feeding input from webcam> -d <choose the device to run the model (default CPU)> -show <select the visualization: win fd fl hp ge crop> -pc <use perf_counts to display performance of layers on each model>
```

**3. Examples of run `the main.py`:**
- Running model with percision **FP32** in **CPU**:
```sh
python main.py -fd "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml" -fl "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml" -hp "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml" -ge "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002.xml" -i "\Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4" -d CPU -show win fd fl hp ge -pc
```

- Running model with percision **FP16** in **CPU**:
```sh
python main.py -fd "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml" -fl "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009.xml" -hp "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml" -ge "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002.xml" -i "\Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4" -d CPU -show win fd fl hp ge -pc
```

- Running model with percision **FP16-INT8** in **CPU**:
```sh
python main.py -fd "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml" -fl "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP16-INT8\landmarks-regression-retail-0009.xml" -hp "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP16-INT8\head-pose-estimation-adas-0001.xml" -ge "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP16-INT8\gaze-estimation-adas-0002.xml" -i "\Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4" -d CPU -show win fd fl hp ge -pc
```

- Running model with percision **FP32** in **CPU** and input through **webcam**
```sh
python main.py -fd "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml" -fl "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml" -hp "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml" -ge "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002.xml" -i cam -d CPU -show win fd fl hp ge -pc
```

## Documentation
_______________

### Command line agruments 
**Try command `python main.py -h` to get help for command line arguments of the application** 
```
E:\Intel-AI\Computer-Pointer-Controller-OpenVINO\src>python main.py -h
usage: main.py [-h] -fd FACE_DETECTION_PATH -fl FACIAL_LANDMARKS_PATH -hp
               HEAD_POSE_PATH -ge GAZE_ESTIMATION_PATH -i INPUT [-pc]
               [-show FLAG_VISUALIZATION [FLAG_VISUALIZATION ...]]
               [-l CPU_EXTENSION] [-d DEVICE] [-pt PROB_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -fd FACE_DETECTION_PATH, --face_detection_path FACE_DETECTION_PATH
                        (required) Path to Face Detection Model .xml file.
  -fl FACIAL_LANDMARKS_PATH, --facial_landmarks_path FACIAL_LANDMARKS_PATH
                        (required) Path to Facial Landmarks Detection Model
                        .xml file.
  -hp HEAD_POSE_PATH, --head_pose_path HEAD_POSE_PATH
                        (required) Path to Head Pose Estimation Model .xml
                        file.
  -ge GAZE_ESTIMATION_PATH, --gaze_estimation_path GAZE_ESTIMATION_PATH
                        (required) Path to Gaze Estimation Model .xml file.
  -i INPUT, --input INPUT
                        (required) Path to Input File either image or video or
                        CAM (using camera).
  -pc, --perf_counts    Report performance countersPrint the real time takes
                        for each layer in the model
  -show FLAG_VISUALIZATION [FLAG_VISUALIZATION ...], --flag_visualization FLAG_VISUALIZATION [FLAG_VISUALIZATION ...]
                        (optional) Visualize the selected model on the output
                        frame window. 'win': display the visualization window
                        (To visualize other model must with 'win'), 'fd': Face
                        Detection Model, 'fl': Facial Landmarks Detection
                        Model, 'crop': Cropped face with Landmarks
                        Detection,'hp': Head Pose Estimation Model, 'ge': Gaze
                        Estimation Model. For example, '--show win fd fl hp ge
                        crop' (Seperate each flag by space).
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        (optional) MKLDNN (CPU)-targeted custom
                        layers.Absolute path to cpu_extension if layers from
                        model are not supported on device.
  -d DEVICE, --device DEVICE
                        (optional) Specify the target device to infer on: CPU,
                        GPU, FPGA or MYRIAD is acceptable. Sample will look
                        for a suitable plugin for device specified (CPU by
                        default)
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        (optional) Probability threshold for detections (0.6
                        by default)
                        
```
## Benchmarks
_______________

- FP32, FP16, FP16-INT8 tested on my CPU: Intel(R) Core(TM)i7-3632QM CPU @ 2.20GHz
- Checked each model load time, average inference time, total load model, total inference time, FPS and model size.

#### Example result of enable `-pc`:
```
## Head pose estimation model performance:
name                                                                   layer_type      exet_type       status          real_time, us
48/Output_0/Data__const123_const_nchw_OIhw8i8o_pool1                   Reorder         jit_uni_FP32    NOT_RUN         0
60/Output_0/Data__const146_const                                       Const           unknown         NOT_RUN         0
60/Output_0/Data__const146_const_nchw_OIhw8i8o_pool3                   Reorder         jit_uni_FP32    NOT_RUN         0
70/Output_0/Data__const114_const                                       Const           unknown         NOT_RUN         0
70/Output_0/Data__const114_const_nchw_Ohwi8o_conv1                     Reorder         jit_uni_FP32    NOT_RUN         0
80/Output_0/Data__const137_const                                       Const           unknown         NOT_RUN         0
80/Output_0/Data__const137_const_nchw_OIhw8i8o_conv3                   Reorder         jit_uni_FP32    NOT_RUN         0
Add_/Fused_Add_                                                        ScaleShift      jit_sse42_FP32  EXECUTED        35
Constant_11352                                                         Const           unknown         NOT_RUN         0
Constant_11352_nchw_OhIw8o4i_conv2                                     Reorder         ref_any_I8      NOT_RUN         0
Constant_11353                                                         Const           unknown         NOT_RUN         0
Constant_11353_nchw_OhIw8o4i_conv4                                     Reorder         ref_any_I8      NOT_RUN         0
Constant_11354                                                         Const           unknown         NOT_RUN         0
Constant_11354_nchw_OhIw8o4i_conv5                                     Reorder         ref_any_I8      NOT_RUN         0
Constant_11355                                                         Const           unknown         NOT_RUN         0
Constant_11355_nchw_OhIw8o4i_conv_fm                                   Reorder         ref_any_I8      NOT_RUN         0
Constant_11356                                                         Const           unknown         NOT_RUN         0
Constant_11356_nchw_OhIw8o4i_angle_y                                   Reorder         ref_any_I8      NOT_RUN         0
Constant_11359                                                         Const           unknown         NOT_RUN         0
Constant_11359_nchw_OhIw8o4i_angle_r/Fused_Add_                        Reorder         ref_any_I8      NOT_RUN         0
Constant_11362                                                         Const           unknown         NOT_RUN         0
Constant_11362_nchw_OhIw8o4i_angle_p                                   Reorder         ref_any_I8      NOT_RUN         0
Constant_12941                                                         Const           unknown         NOT_RUN         0
Constant_12942                                                         Const           unknown         NOT_RUN         0
Constant_12943                                                         Const           unknown         NOT_RUN         0
Constant_12944                                                         Const           unknown         NOT_RUN         0
Constant_12945                                                         Const           unknown         NOT_RUN         0
Constant_12946                                                         Const           unknown         NOT_RUN         0
Constant_12947                                                         Const           unknown         NOT_RUN         0
Constant_12948                                                         Const           unknown         NOT_RUN         0
Constant_12949                                                         Const           unknown         NOT_RUN         0
Constant_12951                                                         Const           unknown         NOT_RUN         0
Constant_12952                                                         Const           unknown         NOT_RUN         0
Constant_12954                                                         Const           unknown         NOT_RUN         0
Constant_12955                                                         Const           unknown         NOT_RUN         0
Constant_12957                                                         Const           unknown         NOT_RUN         0
angle_p                                                                Convolution     jit_sse42_I8    EXECUTED        13
angle_p/WithoutBiases/fq_input_0                                       FakeQuantize    undef           NOT_RUN         0
angle_p_fc                                                             FullyConnected  jit_gemm_FP32   EXECUTED        3
angle_p_fc/WithoutBiases/1_port_transpose3315_const184_const           Const           unknown         NOT_RUN         0
angle_p_fc/flatten_fc_input                                            Reshape         unknown_FP32    NOT_RUN         0
angle_p_fc/flatten_fc_input_ScaleShift_angle_p_fc                      ScaleShift      jit_sse42_FP32  EXECUTED        5
angle_p_nhwc_nchw_angle_p_fc/flatten_fc_input                          Reorder         jit_uni_FP32    EXECUTED        5
angle_r/Fused_Add_                                                     Convolution     jit_sse42_I8    EXECUTED        16
angle_r/Fused_Add__nhwc_nchw_angle_r_fc/flatten_fc_input               Reorder         jit_uni_FP32    EXECUTED        5
angle_r_fc                                                             FullyConnected  jit_gemm_FP32   EXECUTED        4
angle_r_fc/WithoutBiases/1_port_transpose3323_const195_const           Const           unknown         NOT_RUN         0
angle_r_fc/flatten_fc_input                                            Reshape         unknown_FP32    NOT_RUN         0
angle_r_fc/flatten_fc_input_ScaleShift_angle_r_fc                      ScaleShift      jit_sse42_FP32  EXECUTED        6
angle_y                                                                Convolution     jit_sse42_I8    EXECUTED        15
angle_y_fc                                                             FullyConnected  jit_gemm_FP32   EXECUTED        49
angle_y_fc/WithoutBiases/1_port_transpose3319_const210_const           Const           unknown         NOT_RUN         0
angle_y_fc/flatten_fc_input                                            Reshape         unknown_FP32    NOT_RUN         0
angle_y_fc/flatten_fc_input_ScaleShift_angle_y_fc                      ScaleShift      jit_sse42_FP32  EXECUTED        6
angle_y_nhwc_nchw_angle_y_fc/flatten_fc_input                          Reorder         jit_uni_FP32    EXECUTED        6
bn1/variance/Fused_Add_                                                ScaleShift      undef           NOT_RUN         0
bn3/variance/Fused_Add_                                                ScaleShift      undef           NOT_RUN         0
conv1                                                                  Convolution     jit_sse42_FP32  EXECUTED        142
conv2                                                                  Convolution     jit_sse42_I8    EXECUTED        145
conv2/WithoutBiases/fq_input_0                                         FakeQuantize    undef           NOT_RUN         0
conv2_nhwc_nChw8c_pool2                                                Reorder         jit_uni_FP32    EXECUTED        23
conv3                                                                  Convolution     jit_sse42_FP32  EXECUTED        594
conv4                                                                  Convolution     jit_sse42_I8    EXECUTED        89
conv4/WithoutBiases/fq_input_0                                         FakeQuantize    undef           NOT_RUN         0
conv5                                                                  Convolution     jit_sse42_I8    EXECUTED        90
conv_fm                                                                Convolution     jit_sse42_I8    EXECUTED        120
conv_fm/WithoutBiases/fq_input_0                                       FakeQuantize    undef           NOT_RUN         0
data                                                                   Input           unknown         NOT_RUN         0
out_angle_p_fc                                                         Output          unknown_FP32    NOT_RUN         0
out_angle_r_fc                                                         Output          unknown_FP32    NOT_RUN         0
out_angle_y_fc                                                         Output          unknown_FP32    NOT_RUN         0
pool1                                                                  Convolution     jit_sse42_FP32  EXECUTED        190
pool1_FP32_nChw8c_U8_nhwc_conv2                                        Reorder         jit_uni_FP32    EXECUTED        19
pool2                                                                  Pooling         jit_avx_FP32    EXECUTED        26
pool2_ScaleShift_conv3                                                 ScaleShift      jit_sse42_FP32  EXECUTED        8
pool3                                                                  Convolution     jit_sse42_FP32  EXECUTED        189
pool3_FP32_nChw8c_U8_nhwc_conv4                                        Reorder         jit_uni_FP32    EXECUTED        13
pool4                                                                  Pooling         jit_sse42_I8    EXECUTED        7
pool4/fq_input_0                                                       FakeQuantize    undef           NOT_RUN         0
relu_conv1                                                             ReLU            undef           NOT_RUN         0
relu_conv2                                                             ReLU            undef           NOT_RUN         0
relu_conv3                                                             ReLU            undef           NOT_RUN         0
relu_conv4                                                             ReLU            undef           NOT_RUN         0
relu_conv5                                                             ReLU            undef           NOT_RUN         0
relu_p_1                                                               ReLU            undef           NOT_RUN         0
relu_r_1                                                               ReLU            undef           NOT_RUN         0
relu_y_1                                                               ReLU            undef           NOT_RUN         0
Average inference time of HeadPoseEstimation model: 16.855223137035704 ms
```

#### FP32 Project_log 
```
INFO:root:## Face Model path is correct: \Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml
INFO:root:## Landmarks Model path is correct: \Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml
INFO:root:## Headpose Model path is correct: \Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml
INFO:root:## Gaze Model path is correct: \Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002.xml
INFO:root:
Input path exists: \Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4
INFO:root:## Loaded Input Feeder 
INFO:root:FaceDetection load time: 402.233 ms
INFO:root:FacialLandmarkDetection load time: 202.0 ms
INFO:root:HeadPoseEstimation load time: 178.005 ms
INFO:root:GazeEstimation load time: 211.995 ms
INFO:root:Total Load time: 994.233 ms
INFO:root:## All model successfully loaded!
INFO:root:## Start inference on frame!
ERROR:root:flag_return: False. Video has reach to the end...
INFO:root:Average inference time of FaceDetection model: 50.7061845165188 ms
INFO:root:Average inference time of FacialLandmarkDetection model: 1.6961178537142478 ms
INFO:root:Average inference time of HeadPoseEstimation model: 3.2278723635915982 ms
INFO:root:Average inference time of GazeEstimation model: 3.6383079270185053 ms
INFO:root:Total inference time: 38241.687 ms
INFO:root:Total frame: 59
INFO:root:FPS: 1.5428063385806183
ERROR:root:### Camera Stream or Video Stream has reach to the end...###
```

#### FP16 Project_log 
```
INFO:root:## Face Model path is correct: \Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml
INFO:root:## Landmarks Model path is correct: \Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009.xml
INFO:root:## Headpose Model path is correct: \Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml
INFO:root:## Gaze Model path is correct: \Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002.xml
INFO:root:
Input path exists: \Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4
INFO:root:## Loaded Input Feeder 
INFO:root:FaceDetection load time: 378.925 ms
INFO:root:FacialLandmarkDetection load time: 187.5 ms
INFO:root:HeadPoseEstimation load time: 187.506 ms
INFO:root:GazeEstimation load time: 203.12 ms
INFO:root:Total Load time: 957.05 ms
INFO:root:## All model successfully loaded!
INFO:root:## Start inference on frame!
ERROR:root:flag_return: False. Video has reach to the end...
INFO:root:Average inference time of FaceDetection model: 46.96024070351811 ms
INFO:root:Average inference time of FacialLandmarkDetection model: 0.5190655336541644 ms
INFO:root:Average inference time of HeadPoseEstimation model: 2.277273242756472 ms
INFO:root:Average inference time of GazeEstimation model: 3.472477702771203 ms
INFO:root:Total inference time: 36004.365 ms
INFO:root:Total frame: 59
INFO:root:FPS: 1.6387068103544051
ERROR:root:### Camera Stream or Video Stream has reach to the end...###
```
#### FP16-INT8 Project_log 
```
INFO:root:## Face Model path is correct: \Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml
INFO:root:## Landmarks Model path is correct: \Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP16-INT8\landmarks-regression-retail-0009.xml
INFO:root:## Headpose Model path is correct: \Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP16-INT8\head-pose-estimation-adas-0001.xml
INFO:root:## Gaze Model path is correct: \Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP16-INT8\gaze-estimation-adas-0002.xml
INFO:root:
Input path exists: \Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4
INFO:root:## Loaded Input Feeder 
INFO:root:FaceDetection load time: 328.122 ms
INFO:root:FacialLandmarkDetection load time: 218.754 ms
INFO:root:HeadPoseEstimation load time: 312.502 ms
INFO:root:GazeEstimation load time: 365.017 ms
INFO:root:Total Load time: 1224.395 ms
INFO:root:## All model successfully loaded!
INFO:root:## Start inference on frame!
ERROR:root:flag_return: False. Video has reach to the end...
INFO:root:Average inference time of FaceDetection model: 46.5065099425235 ms
INFO:root:Average inference time of FacialLandmarkDetection model: 0.9639626842434124 ms
INFO:root:Average inference time of HeadPoseEstimation model: 2.110372155399646 ms
INFO:root:Average inference time of GazeEstimation model: 4.738435906878973 ms
INFO:root:Total inference time: 35699.297 ms
INFO:root:Total frame: 59
INFO:root:FPS: 1.6527073587495449
ERROR:root:### Camera Stream or Video Stream has reach to the end...###
```

### Comparison model size and load time
|Model|Type|Size|Load time|Average inference time|
| ------ | ------ | ------ |------ |------ |
|face-detection-adas-binary-0001|FP32-INT1|1.86 MB|402.233 ms|50.7061845165188 ms|
||FP32-INT1|1.86 MB|378.925 ms|46.96024070351811 ms|
||FP32-INT1|1.86 MB|328.122 ms|46.5065099425235 ms|
|landmarks-regression-retail-0009|FP32|786 KB|202.0 ms|1.6961178537142478 ms|
||FP16|413 KB|187.5 ms|0.5190655336541644 ms|
||FP16-INT8|314 KB|218.754 ms|0.9639626842434124 ms|
|head-pose-estimation-adas-0001 |FP32|7.34 MB|178.005 ms|3.2278723635915982 ms|
||FP16|3.69 MB|187.506 ms|2.277273242756472 ms|
||FP16-INT8|2.05 MB|312.502 ms|2.110372155399646 ms|
|gaze-estimation-adas-0002|FP32|7.24 MB|211.995 ms|3.6383079270185053 ms|
||FP16|3.69 MB|203.12 ms|3.472477702771203 ms|
||FP16-INT8|2.09 MB|365.017 ms|4.738435906878973 ms|

### Comparison of total load and inference time
||Total Model Load time|Total Inference Time|FPS|
| ------ | ------ | ------ | ------ |
|FP32|994.233 ms|38241.687 ms|1.5428063385806183|
|FP16|957.05 ms|36004.365 ms|1.6387068103544051|
|FP16-INT8|1224.395 ms|35699.297 ms|1.6527073587495449|

## Results
_______________
- For different precision, the model size decreases follow the order of FP32 > FP16 > FP16-INT8. The total inference time follows the same order in this case, the INT8 is faster than FP16 and FP32 is slower than FP32. Lower precision model uses less memory. However, remember lower precision of model also lose the accuracy of model.
- Memory Access of FP16 is half the size compared with FP32, which reduces memory usage of a neural network. FP16 data transfers are faster than FP32, which improves speed (TFLOPS) and performance.
- The Model load time and FPS of them are almost the same, but Total model load time of FP16 is much faster than others. For FPS INT-8 is better than others. Besides model only load only when model initializes.
- In order to achieving the most reasonable combination, we do not want too longer inference time also too low accuracy. Moreover, in some specific scenario such as low budget. We do not want to waste storage attempt to get very high accuracy. To achieve the balance, we need to consider the volume of sacrifice.
- I would choose FP16 as best option based on all results, for each model the load time, average inference time, total load time and total inference time in FP16 gives very good results.

## Stand Out Suggestions
_______________
- Use the VTune Amplifier to find hotspots in inference engine pipline.
- Build an inference pipeline for both video file and webcam feed as input. Allow the user to select their input option in the command line arguments. 
- Benchmark the running times of different parts of the preprocessing and inference pipeline and let the user specify a CLI argument if they want to see the benchmark timing. Use the get_perf_counts API to print the time it takes for each layer in the model. 

### Edge Cases
_______________
- If there will multiple face situation, the model only extracts one face to control the mouse pointer and ignore the other faces.
- If due to certain reason, model couldn't detect the face. It will continue process another frame untill it detects face or keyboard interrupt to exit the program.

