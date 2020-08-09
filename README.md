# Computer Pointer Controller
https://youtu.be/ZJ8y--zcBag
*TODO:* Write a short introduction to your project

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

## Demo
*TODO:* Explain how to run a basic demo of your model.
FP32
```sh
python main.py -f "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml" -fl "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml" -hp "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml" -g "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002.xml" -i "\Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4" -d CPU -flags fd fld hp ge
```
FP16
```sh
python main.py -f "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml" -fl "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009.xml" -hp "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml" -g "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002.xml" -i "\Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4" -d CPU -flags fd fld hp ge
```
FP16-INT8
```sh
python main.py -f "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml" -fl "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP16-INT8\landmarks-regression-retail-0009.xml" -hp "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP16-INT8\head-pose-estimation-adas-0001.xml" -g "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP16-INT8\gaze-estimation-adas-0002.xml" -i "\Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4" -d CPU -flags fd fld hp ge
```
## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.
FP32
```
INFO:root:Model Load time: 0.9885485172271729
INFO:root:Inference time: 28.26241898536682
INFO:root:FPS: 2.0877565463552723
ERROR:root:VideoStream ended...
```
FP16
```
INFO:root:Model Load time: 0.9749979972839355
INFO:root:Inference time: 28.06306004524231
INFO:root:FPS: 2.1026372059871705
ERROR:root:VideoStream ended...
```
FP16-INT8
```
INFO:root:Model Load time: 1.3344995975494385
INFO:root:Inference time: 28.29212784767151
INFO:root:FPS: 2.0855425945563804
ERROR:root:VideoStream ended...
```
## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

