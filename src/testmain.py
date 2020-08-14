import os
import cv2
import numpy as np
import time
import logging
from argparse import ArgumentParser
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FacialLandmarkDetection
from gaze_estimation import GazeEstimation


def main():
	'''
	Grab command line args
	'''
	args = build_argparser().parse_args()
	preview_flags = args.preview_flags
	File_path_input = args.input
	inputFeeder = None

	# create a log file
	logging.basicConfig(filename='Project_log.log', level=logging.INFO)
	logger = logging.getLogger()


	# create dict get model selected in command 
	Dict_model_path = {
				'FaceDetection':args.facedetection, 
				'FacialLandmarkDetection':args.faciallandmark,
				'HeadPoseEstimation':args.headpose, 
				'GazeEstimation':args.gazeestimation
	}

	if File_path_input.lower()=="cam":
		inputFeeder = InputFeeder("cam")
	else:
		if not os.path.isfile(File_path_input):
			print("Cannot find video file, please check once again!")
			logger.error("Unable to find specified video file, Please check once again!")
			exit(1)
		inputFeeder = InputFeeder("video", File_path_input)

	# check if the file exists
	for model_key in Dict_model_path.keys():
		if not os.path.isfile(Dict_model_path[model_key]):
			print("Cannot find " + model_key + ' Please enter again!')
			logger.error("Unable to find specified "+ model_key +" xml file, Please check once again!")
			exit(1)

	
	mouse_handler = MouseController('medium', 'fast')

	# initialize face detection mode
	model_fd = FaceDetection(Dict_model_path['FaceDetection'], args.device, args.cpu_extension)
	# initialize facial landmarks detection model
	model_fld = FacialLandmarkDetection(Dict_model_path['FacialLandmarkDetection'], args.device, args.cpu_extension)
	# initialize head pose estimation model
	model_hpe = HeadPoseEstimation(Dict_model_path['HeadPoseEstimation'], args.device, args.cpu_extension)
	# initialize gaze estimation model
	model_gem = GazeEstimation(Dict_model_path['GazeEstimation'], args.device, args.cpu_extension)
	


	inputFeeder.load_data()
	# load face detection model
	model_fd_start_time = time.time()
	model_fd.load_model()
	model_fd_load_time = time.time() - model_fd_start_time
	logger.info('FaceDetection load time: ' + str(round(model_fd_load_time*1000, 3)) + ' ms')

	# load facial landmarks detection model
	model_fld_start_time = time.time()
	model_fld.load_model()
	model_fld_load_time = time.time() - model_fld_start_time
	logger.info('FacialLandmarkDetection load time: ' + str(round(model_fld_load_time*1000, 3)) + ' ms')

	# load head pose estimation model
	model_hpe_start_time = time.time()
	model_hpe.load_model()
	model_hpe_load_time = time.time() - model_hpe_start_time
	logger.info('HeadPoseEstimation load time: ' + str(round(model_hpe_load_time*1000, 3)) + ' ms')

	# load gaze estimation model
	model_gem_start_time = time.time()
	model_gem.load_model()
	model_gem_load_time, total_load_time = (time.time() - model_gem_start_time), (time.time() - model_fd_start_time)
	logger.info('GazeEstimation load time: ' + str(round(model_gem_load_time*1000, 3)) + ' ms')

	# Model load time in total 
	logger.info('Model Total Load time: ' + str(round(total_load_time*1000, 3)) + ' ms')

	frame_count = 0
	start_infer_time = time.time()

	for flag_return, frame in inputFeeder.next_batch():
		# print(flag_return)
		# check if reach to the end, return true or false
		if not flag_return:
			print('\n Video has reach to the end....')
			break

		event_key = cv2.waitKey(60)

		frame_count += 1

		# each 1 frame update the display window
		if frame_count % 1 == 0:
			cv2.imshow('Video Stream', cv2.resize(frame, (900,700)))
		
		# face detection model
		cropped_face, face_coords = model_fd.predict(frame.copy(), args.prob_threshold)

		if type(cropped_face)==int:
			
			print('\nCannot detect face!')
			logger.error("Unable to detect the face. Please check once again!")
			# Esc key is 27
			if event_key==27:
				break
			continue

		# facial landmarks detection model
		left_eye_box, right_eye_box, eyes_coords = model_fld.predict(cropped_face.copy())
		
		# head pose estimation model
		hpe_output = model_hpe.predict(cropped_face.copy())

		# gaze estimation model
		# needs input of left eye & right eye & head pose
		output_mouse_position, gaze_vector = model_gem.predict(left_eye_box, right_eye_box, hpe_output)

		# if the preview flags is activated
		if (not len(preview_flags)==0):

			frame_copy = frame.copy()

			if 'fd' in preview_flags:

				if len(preview_flags) == 1:
					cv2.rectangle(frame_copy, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 2) 
					# cv2.imshow("Face detection",cv2.resize(frame_copy,(800,600)))
					# frame_copy = cropped_face
				elif (len(preview_flags) == 1) and File_path_input.lower()=="cam":
					cv2.rectangle(frame_copy, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 2)
				else:
					# print(cropped_face.shape)
					frame_copy = cropped_face

			if 'fld' in preview_flags:
				frame_copy = cropped_face
				cv2.rectangle(frame_copy, (eyes_coords[0][0]-10, eyes_coords[0][1]-10), (eyes_coords[0][2]+10, eyes_coords[0][3]+10), (255,255,255), 2)
				cv2.rectangle(frame_copy, (eyes_coords[1][0]-10, eyes_coords[1][1]-10), (eyes_coords[1][2]+10, eyes_coords[1][3]+10), (255,255,255), 2)
                

			if 'hp' in preview_flags:
				frame_copy = cropped_face
				cv2.putText(frame_copy, "Angles of Head Pose:", (10, 10), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)
				cv2.putText(frame_copy, "Yaw: {:.1f}".format(hpe_output[0]), (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)
				cv2.putText(frame_copy, "Pitch: {:.1f}".format(hpe_output[1]), (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)
				cv2.putText(frame_copy, "Roll: {:.1f}".format(hpe_output[2]), (10, 55), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)

			if 'ge' in preview_flags:				
				
				w_len, x, y,  = 200, int(gaze_vector[0]*12), int(gaze_vector[1]*12)
				frame_copy = cropped_face

				line_left_eye =cv2.arrowedLine(left_eye_box.copy(), (x-w_len, y-w_len), (x+w_len, y+w_len), (0,255,0), 2)
				cv2.arrowedLine(line_left_eye, (x-w_len, y+w_len), (x+w_len, y-w_len), (0,255,0), 2)

				line_right_eye = cv2.arrowedLine(right_eye_box.copy(), (x-w_len, y-w_len), (x+w_len, y+w_len), (0,255,0), 2)
				cv2.arrowedLine(line_right_eye, (x-w_len, y+w_len), (x+w_len, y-w_len), (0,255,0), 2)
				
				# print(line_left_eye.shape, line_right_eye.shape)
				cropped_face[eyes_coords[0][1]:eyes_coords[0][3],eyes_coords[0][0]:eyes_coords[0][2]] = line_left_eye
				cropped_face[eyes_coords[1][1]:eyes_coords[1][3],eyes_coords[1][0]:eyes_coords[1][2]] = line_right_eye
                

			cv2.imshow("Visualization of option {}".format(preview_flags), cv2.resize(frame_copy,(700,600)))

		if frame_count % 5==0:
			mouse_handler.move(output_mouse_position[0], output_mouse_position[1])

		if event_key==27:
			print("\nUser keyboard exit!....")
			break

	# calculate the total inference time and fps
	total_infer_time = time.time() - start_infer_time
	fps = frame_count / round(total_infer_time, 3)

	# loggging into project log file
	logger.info('Inference time: ' + str(round(total_infer_time*1000, 3)) + ' ms')
	logger.info('FPS: ' + str(fps))
	logger.error("### Camera Stream or Video Stream has reach to the end...###")

	cv2.destroyAllWindows()
	inputFeeder.close()


def build_argparser():
	'''
	Parser of command line arguments.

	Example:
	python main.py -f "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml" \
	-fl "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml" \
	-hp "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml" \
	-g "\Intel-AI\Computer-Pointer-Controller-OpenVINO\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002.xml" \
	-i "\Intel-AI\Computer-Pointer-Controller-OpenVINO\bin\demo.mp4" \
	-d CPU -flags fd fld hp ge

	return: command line arguments
	'''
	parser = ArgumentParser()
	# parser of face detection path
	parser.add_argument("-f", "--facedetection", required=True, type=str, 
						help="Specify the Path to Face Detection model .xml file.")
	# parser of landmark detection path
	parser.add_argument("-fl", "--faciallandmark", required=True, type=str, 
						help="Specify the Path to Facial Landmark Detection model .xml file.")
	# parser of headpose detection path
	parser.add_argument("-hp", "--headpose", required="True", type=str , 
						help="Specify the Path to Head Pose Estimation model .xml file.")
	# parser of gaze detection path
	parser.add_argument("-g", "--gazeestimation", required=True, type=str , 
						help="Specify the Path to Gaze Estimation model .xml file.")
	# parser of input path
	parser.add_argument("-i", "--input", required=True, type=str , 
						help="Specify the Path to either video file or enter 'cam' for accessing the webcam")
	# parser of flags which choose way of visualization
	parser.add_argument("-flags", "--preview_flags", required=False, nargs='+', default=[], 
						help="Specify the flags from fd, fld, hp, ge like '--flags fd hp fld ge' (Seperate each flag by space)"
                             "to show the visualization of different model outputs of each frame," 
                             "fd represents Face Detection, fld represents Facial Landmark Detection"
                             "hp represents Head Pose Estimation, ge represents Gaze Estimation." )
	# parser of cpu_extension
	parser.add_argument("-l", "--cpu_extension", required=False, type=str , default=None,
						help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
	# # parser of threshold
	parser.add_argument("-prob", "--prob_threshold", required=False, type=float, default=0.6, 
						help="Probability threshold for model to accurately detect the face from the video frame."
							  "(0.6 by default)")
	# parser of device
	parser.add_argument("-d", "--device", type=str, default="CPU", 
						help="Specify the target device to perform the model inference: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable."                            
                             "(CPU by default)")
	
	return parser


if __name__ == '__main__':
	main()