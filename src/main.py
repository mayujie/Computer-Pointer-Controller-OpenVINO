import cv2
import os
import logging
import numpy as np
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarkDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder
import time


def main():
	'''
	Grab command line args
	'''
	args = build_argparser().parse_args()
	preview_flags = args.preview_flags

	logging.basicConfig(filename='Project_log.log', level=logging.INFO)
	logger = logging.getLogger()
	File_path_input = args.input
	inputFeeder = None

	if File_path_input.lower()=="cam":
		inputFeeder = InputFeeder("cam")
	else:
		if not os.path.isfile(File_path_input):
			logger.error("Unable to find specified video file, Please check once again!")
			exit(1)
		inputFeeder = InputFeeder("video", File_path_input)

	Dict_model_path = {'FaceDetection':args.facedetection, 'FacialLandmarkDetection':args.faciallandmark,
	'HeadPoseEstimation':args.headpose, 'GazeEstimation':args.gazeestimation}

	for fileNameKey in Dict_model_path.keys():
		if not os.path.isfile(Dict_model_path[fileNameKey]):
			logger.error("Unable to find specified "+ fileNameKey +" xml file, Please check once again!")
			exit(1)

	
	mouse_handler = MouseController('medium', 'fast')
	model_fd = FaceDetection(Dict_model_path['FaceDetection'], args.device, args.cpu_extension)
	model_fld = FacialLandmarkDetection(Dict_model_path['FacialLandmarkDetection'], args.device, args.cpu_extension)
	model_gem = GazeEstimation(Dict_model_path['GazeEstimation'], args.device, args.cpu_extension)
	model_hpe = HeadPoseEstimation(Dict_model_path['HeadPoseEstimation'], args.device, args.cpu_extension)

	inputFeeder.load_data()

	model_fd_start_time = time.time()
	model_fd.load_model()
	model_fd_load_time = time.time() - model_fd_start_time

	model_fld_start_time_ = time.time()
	model_fld.load_model()
	model_fld_load_time = time.time() - model_fld_start_time_

	model_hpe_start_time = time.time()
	model_hpe.load_model()
	model_hpe_load_time = time.time() - model_hpe_start_time

	model_gem_start_time = time.time()
	model_gem.load_model()
	model_gem_load_time, total_load_time = (time.time() - model_gem_start_time), (time.time() - model_fd_start_time)


	logger.info('FaceDetection load time: ' + str(round(model_fd_load_time*1000, 3)) + ' ms')
	logger.info('FacialLandmarkDetection load time: ' + str(round(model_fld_load_time*1000, 3)) + ' ms')
	logger.info('HeadPoseEstimation load time: ' + str(round(model_hpe_load_time*1000, 3)) + ' ms')
	logger.info('GazeEstimation load time: ' + str(round(model_gem_load_time*1000, 3)) + ' ms')
	logger.info('Model Load time: ' + str(round(total_load_time*1000, 3)) + ' ms')

	frame_count = 0
	start_infer_time = time.time()

	for ret, frame in inputFeeder.next_batch():
		if not ret:
			break
		frame_count+=1
		if frame_count %5 == 0:
			cv2.imshow('Video Stream', cv2.resize(frame, (900,700)))

		key = cv2.waitKey(60)
		# face detection model
		cropped_face, face_coords = model_fd.predict(frame.copy(), args.prob_threshold)

		if type(cropped_face)==int:
			logger.error("Unable to detect the face. Please check once again!")
			# Esc key is 27
			if key==27:
				break
			continue

		# head pose estimation model
		hp_out = model_hpe.predict(cropped_face.copy())
		# facial landmarks detection model
		left_eye, right_eye, eye_coords = model_fld.predict(cropped_face.copy())
		# gaze estimation model
		# needs input of left eye & right eye & head pose
		new_mouse_coord, gaze_vector = model_gem.predict(left_eye, right_eye, hp_out)

		if (not len(preview_flags)==0):
			preview_frame = frame.copy()

			if 'fd' in preview_flags:
				# cv2.rectangle(preview_frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 3) 
				preview_frame = cropped_face

			if 'fld' in preview_flags:
				cv2.rectangle(cropped_face, (eye_coords[0][0]-15, eye_coords[0][1]-15), (eye_coords[0][2]+15, eye_coords[0][3]+15), (255,255,255), 2)
				cv2.rectangle(cropped_face, (eye_coords[1][0]-15, eye_coords[1][1]-15), (eye_coords[1][2]+15, eye_coords[1][3]+15), (255,255,255), 2)
                # preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = cropped_face

			if 'hp' in preview_flags:
				cv2.putText(preview_frame, "Pose Angles:  yaw: {:.2f} | pitch: {:.2f} | roll: {:.2f}".format(hp_out[0],hp_out[1],hp_out[2]), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.25, (0, 255, 255), 1)
			
			if 'ge' in preview_flags:
				x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160

				le =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (0,255,0), 2)
				cv2.line(le, (x-w, y+w), (x+w, y-w), (0,255,0), 2)

				re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (0,255,0), 2)
				cv2.line(re, (x-w, y+w), (x+w, y-w), (0,255,0), 2)
				
				cropped_face[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]] = le
				cropped_face[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]] = re
                # preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = cropped_face

			cv2.imshow("Visualization of cropped face",cv2.resize(preview_frame,(600,600)))

		if frame_count % 5==0:
			mouse_handler.move(new_mouse_coord[0], new_mouse_coord[1])

		if key==27:
			break

	total_infer_time = time.time() - start_infer_time
	fps = frame_count / round(total_infer_time, 3)

	logger.info('Inference time: ' + str(round(total_infer_time*1000, 3)) + ' ms')
	logger.info('FPS: ' + str(fps))

	logger.error("### VideoStream has reach to the end...")
	cv2.destroyAllWindows()
	inputFeeder.close()


def build_argparser():
	'''
	Parse command line arguments.
	return: command line arguments

	Example:
	python main.py -f "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml" \
	-fl "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml" \
	-hp "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml" \
	-g "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002.xml" \
	-i "\Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4" \
	-d CPU -flags fd fld hp ge
	'''
	parser = ArgumentParser()
	parser.add_argument("-f", "--facedetection", required=True, type=str, 
						help="Specify the Path to Face Detection model .xml file.")
	parser.add_argument("-fl", "--faciallandmark", required=True, type=str, 
						help="Specify the Path to Facial Landmark Detection model .xml file.")
	parser.add_argument("-hp", "--headpose", required="True", type=str , 
						help="Specify the Path to Head Pose Estimation model .xml file.")
	parser.add_argument("-g", "--gazeestimation", required=True, type=str , 
						help="Specify the Path to Gaze Estimation model .xml file.")
	parser.add_argument("-i", "--input", required=True, type=str , 
						help="Specify the Path to either video file or enter 'cam' for accessing the webcam")
	parser.add_argument("-flags", "--preview_flags", required=False, nargs='+', 
						default=[], 
						help="Specify the flags from fd, fld, hp, ge like '--flags fd hp fld ge' (Seperate each flag by space)"
                             "to show the visualization of different model outputs of each frame," 
                             "fd represents Face Detection, fld represents Facial Landmark Detection"
                             "hp represents Head Pose Estimation, ge represents Gaze Estimation." )
	parser.add_argument("-l", "--cpu_extension", required=False, type=str , 
						default=None,
						help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
	parser.add_argument("-prob", "--prob_threshold", required=False, type=float, 
						default=0.6, 
						help="Probability threshold for model to accurately detect the face from the video frame."
							  "(0.6 by default)")
	parser.add_argument("-d", "--device", type=str, default="CPU", 
						help="Specify the target device to perform the model inference: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable."                            
                             "(CPU by default)")
	
	return parser


if __name__ == '__main__':
	main()