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

'''
Used reference resources below. 
I have commented what is the result from each model in each part of code. And how to proceed next.

Reference resources:
https://knowledge.udacity.com/questions/254779
https://knowledge.udacity.com/questions/171017
https://knowledge.udacity.com/questions/257811
https://gist.github.com/justinshenk/9917891c0433f33967f6e8cd8fcaa49a
https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1InferRequest.html#details
'''

def main():
	## calling argparser
	args = build_argparser().parse_args()
	# create a log file
	logging.basicConfig(filename='Project_log.log', level=logging.INFO)
	logger = logging.getLogger()

	## get args input variable 
	input_path = args.input
	## get args visualization flags
	visual_flags = args.flag_visualization
	## put all path of model from args in to dict
	Dict_model_path = {
		'Face': args.face_detection_path,
		'Landmarks': args.facial_landmarks_path,
		'Headpose': args.head_pose_path,
		'Gaze': args.gaze_estimation_path
	}

	## put all keys for visualization in dict
	Dict_visual_keys = {
		'args_face': 'fd',
		'args_land': 'fl',
		'args_head': 'hp',
		'args_gaze': 'ge',
		'args_crop': 'crop',
		'args_win': 'win'
	}


	## check if model exists in given path
	for model_key in Dict_model_path.keys():
		print(Dict_model_path[model_key])
		if not os.path.isfile(Dict_model_path[model_key]):
			print("\n## " + model_key + " Model path not exists: " + Dict_model_path[model_key] + ' Please try again !!!')
			logger.error("## " + model_key + " Model path not exists: " + Dict_model_path[model_key] + ' Please try again !!!')
			exit(1)
		else:
			print('## '+model_key + " Model path is correct: " + Dict_model_path[model_key])
			logger.info('## '+model_key + " Model path is correct: " + Dict_model_path[model_key])


	## check if using CAMERA or video file or image
	if input_path == "CAM" or input_path=="cam":
		print("\n## You are using CAMERA right now..." + input_path + " detected!")
		logger.info("\n## You are using CAMERA right now..." + input_path + " detected!")
		feeder_in = InputFeeder(input_path.lower())
	else:
		## check if input file exists in given path
		if not os.path.isfile(input_path):
			print("\nInput file not exists in Path: " + input_path + ". Please check again !!!")
			logger.error("## Input file not exists in Path: " + input_path + ". Please check again !!!")
			exit(1)
		else:
			print('\nInput path exists: '+ input_path)
			logger.info('\nInput path exists: '+ input_path)
			feeder_in = InputFeeder("video", input_path)


	## handler for mouse moving by precision and speed
	mouse_handler = MouseController('medium', 'fast')

	## initialize face detection mode
	model_fd = FaceDetection(Dict_model_path['Face'], args.device, args.cpu_extension)
	## initialize facial landmarks detection model
	model_fld = FacialLandmarkDetection(Dict_model_path['Landmarks'], args.device, args.cpu_extension)
	## initialize head pose estimation model
	model_hpe = HeadPoseEstimation(Dict_model_path['Headpose'], args.device, args.cpu_extension)
	## initialize gaze estimation model
	model_ge = GazeEstimation(Dict_model_path['Gaze'], args.device, args.cpu_extension)


	feeder_in.load_data()
	print("## Loaded Input Feeder ")
	logger.info("## Loaded Input Feeder ")

	## load face detection model
	model_fd_start_time = time.time()
	model_fd.load_model()
	model_fd_load_time = (time.time() - model_fd_start_time)*1000
	logger.info('FaceDetection load time: ' + str(round(model_fd_load_time, 3)) + ' ms')

	## load facial landmarks detection model
	model_fld_start_time = time.time()
	model_fld.load_model()
	model_fld_load_time = (time.time() - model_fld_start_time)*1000
	logger.info('FacialLandmarkDetection load time: ' + str(round(model_fld_load_time, 3)) + ' ms')

	## load head pose estimation model
	model_hpe_start_time = time.time()
	model_hpe.load_model()
	model_hpe_load_time = (time.time() - model_hpe_start_time)*1000
	logger.info('HeadPoseEstimation load time: ' + str(round(model_hpe_load_time, 3)) + ' ms')

	## load gaze estimation model
	model_ge_start_time = time.time()
	model_ge.load_model()
	model_ge_load_time, total_load_time = (time.time() - model_ge_start_time)*1000, (time.time() - model_fd_start_time)*1000
	logger.info('GazeEstimation load time: ' + str(round(model_ge_load_time, 3)) + ' ms')
	## Model load time in total 
	logger.info('Total Load time: ' + str(round(total_load_time, 3)) + ' ms')

	print('\n## All model successfully loaded!')
	logger.info('## All model successfully loaded!')

	frame_count = 0
	print("## Start inference on frame!")
	logger.info("## Start inference on frame!")
	
	## empty list for each model to accumulate infer time and later get average infer time
	fd_infer_time = []
	fld_infer_time = []
	hpe_infer_time = []
	ge_infer_time = []

	start_infer_time = time.time()
	## loop through each frame and start inference on each model
	for flag_return, frame in feeder_in.next_batch():
		# print(flag_return)
		if not flag_return:
			print('\nflag_return: ' + str(flag_return) + '. Video has reach to the end...')
			logger.error('flag_return: ' + str(flag_return) + '. Video has reach to the end...')
			break

		event_key = cv2.waitKey(60)
		## frame count add by 1
		frame_count += 1
		print('\nNo. frame: {}'.format(frame_count))

		if event_key ==27:
			print("\nUser keyboard exit!....")
			break

		## Face detection ##
		t0 = time.time()
		cropped_face, face_coords = model_fd.predict(frame.copy(), args.prob_threshold, args.perf_counts)
		# print(cropped_face.shape)
		## face_coords 
		## top left, bottom right
		fd_infer_time.append((time.time() - t0)*1000)
		# print(fd_infer_time)
		print("Average inference time of FaceDetection model: {} ms".format(np.average(np.asarray(fd_infer_time))))
		
		## if no face detected
		if len(face_coords)==0:
			print("## No Face detected...")
			logger.error("## No face detected. Please check once again!")
			continue
		
		## Landmarks detection ##
		t1 = time.time()
		l_eye_box, r_eye_box, eyes_coords = model_fld.predict(cropped_face.copy(), args.perf_counts)
		# print(l_eye_box.shape, r_eye_box.shape) # left eye and right eye image
		## [left eye box, right eye box] 
		## [[leye_xmin,leye_ymin,leye_xmax,leye_ymax], [reye_xmin,reye_ymin,reye_xmax,reye_ymax]]
		# print(eyes_coords)
		fld_infer_time.append((time.time()- t1)*1000)
		# print(fld_infer_time)
		print("Average inference time of FacialLandmarkDetection model: {} ms".format(np.average(np.asarray(fld_infer_time))))
		
		
		## Head pose detection ##
		t2 = time.time()
		hpe_output = model_hpe.predict(cropped_face.copy(), args.perf_counts)
		# [6.927431583404541, -4.0265960693359375, -1.8397517204284668]
		# print(hpe_output) # yaw, pitch, roll
		hpe_infer_time.append((time.time() - t2)*1000)
		print("Average inference time of HeadPoseEstimation model: {} ms".format(np.average(np.asarray(hpe_infer_time))))

		## Gaze estimation ##		
		t3 = time.time()
		mouse_position, gaze_vector = model_ge.predict(l_eye_box, r_eye_box, hpe_output, args.perf_counts)
		## mouse position (x, y), gaze_vector [-0.13984774, -0.38296703, -0.9055522 ]
		ge_infer_time.append((time.time() - t3)*1000)
		print("Average inference time of GazeEstimation model: {} ms".format(np.average(np.asarray(ge_infer_time))))

		# print('@@@@@@@@@@@@@', len(visual_flags))
				
		## Visualize the result if visual_flags activated
		if len(visual_flags) > 0 and len(visual_flags) <= 6 and Dict_visual_keys['args_win'] in visual_flags:
			frame_copy = frame.copy()

			if Dict_visual_keys['args_face'] in visual_flags:
				# Face
				cv2.rectangle(frame_copy, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 2) 				
				
			if Dict_visual_keys['args_land'] in visual_flags:
				# Facial Landmark left right eyes
				cv2.rectangle(frame_copy, (face_coords[0] + eyes_coords[0][0], face_coords[1] + eyes_coords[0][1]), (face_coords[0]+eyes_coords[0][2], face_coords[1]+eyes_coords[0][3]),(255,255,255), 2)
				cv2.rectangle(frame_copy, (face_coords[0] + eyes_coords[1][0], face_coords[1] + eyes_coords[1][1]), (face_coords[0]+eyes_coords[1][2], face_coords[1]+eyes_coords[1][3]),(255,255,255), 2)				
			
			if Dict_visual_keys['args_crop'] in visual_flags:
				## cropped face with landmarks left and right eyes ##
				land_frame = cropped_face.copy()
				cv2.rectangle(land_frame, (eyes_coords[0][0], eyes_coords[0][1]), (eyes_coords[0][2],eyes_coords[0][3]),(0,255,0), 2)
				cv2.rectangle(land_frame, (eyes_coords[1][0], eyes_coords[1][1]), (eyes_coords[1][2],eyes_coords[1][3]),(0,255,0), 2)
				cv2.imshow('FacialLandmark', cv2.resize(land_frame, (300, 400)))

			if Dict_visual_keys['args_head'] in visual_flags:
				# Head Pose values
				cv2.putText(frame_copy, "Angles of Head Pose:", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
				cv2.putText(frame_copy, "Yaw: {:.2f}".format(hpe_output[0]), (10, 55), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
				cv2.putText(frame_copy, "Pitch: {:.2f}".format(hpe_output[1]), (10, 85), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
				cv2.putText(frame_copy, "Roll: {:.2f}".format(hpe_output[2]), (10, 115), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

			if Dict_visual_keys['args_gaze'] in visual_flags:
				# Gaze arrow left right eyes
				x, y = gaze_vector[0:2]
				len_add = 400
				## eye left center point (x, y)
				eye_left_center = (int(face_coords[0] + (eyes_coords[0][0]+eyes_coords[0][2])/2), int(face_coords[1] + (eyes_coords[0][1]+eyes_coords[1][3])/2))
				## eye right center point (x, y)
				eye_right_center = (int(face_coords[0] + (eyes_coords[1][0]+eyes_coords[1][2])/2), int(face_coords[1] + (eyes_coords[1][1]+eyes_coords[1][3])/2))			
				## draw arrow line for both gaze of eyes
				cv2.arrowedLine(frame_copy, eye_left_center, (int(eye_left_center[0]+x*len_add), int(eye_left_center[1]-y*len_add)), (0,0,255), 3)
				cv2.arrowedLine(frame_copy, eye_right_center, (int(eye_right_center[0]+x*len_add), int(eye_right_center[1]-y*len_add)), (0,0,255), 3)
			
			## if with '-show win' without model keys will only display normal video stream
			cv2.imshow('Visualization', cv2.resize(frame_copy, (800,700)))
		else:
			print("\n## No Visualization, Only information displaying... \n## If needs visualization please add '-show' with specific keys...")


		if frame_count % 4 == 0:
			## start move mouse each 4 frames
			mouse_handler.move(mouse_position[0], mouse_position[1])

	total_infer_time = time.time() - start_infer_time
	fps = frame_count / round(total_infer_time, 3)

	print('Total inference time: ' + str(round(total_infer_time*1000, 3)) + ' ms')
	print("Total frame: " + str(frame_count))
	print('FPS: ' + str(fps))

	## loggging into project log file
	# logger.info('Total inference time: ' + str(round(total_infer_time, 3)) + ' s')	
	logger.info("Average inference time of FaceDetection model: {} ms".format(np.average(np.asarray(fd_infer_time))))
	logger.info("Average inference time of FacialLandmarkDetection model: {} ms".format(np.average(np.asarray(fld_infer_time))))
	logger.info("Average inference time of HeadPoseEstimation model: {} ms".format(np.average(np.asarray(hpe_infer_time))))
	logger.info("Average inference time of GazeEstimation model: {} ms".format(np.average(np.asarray(ge_infer_time))))
	logger.info('Total inference time: ' + str(round(total_infer_time*1000, 3)) + ' ms')
	logger.info("Total frame: " + str(frame_count))
	logger.info('FPS: ' + str(fps))
	logger.error("### Camera Stream or Video Stream has reach to the end...###")

	cv2.destroyAllWindows()
	feeder_in.close()


def build_argparser():
	'''
	parse arguments for main.py to active some specific function.
	return the args values pass to needed function
	
	'''
	parser = ArgumentParser()
	## parser of face detection path
	parser.add_argument("-fd", "--face_detection_path", required=True, type=str, 
		help="(required) Path to Face Detection Model .xml file.")
	## parser of landmark detection path
	parser.add_argument("-fl", "--facial_landmarks_path", required=True, type=str, 
		help="(required) Path to Facial Landmarks Detection Model .xml file.")
	## parser of headpose detection path
	parser.add_argument("-hp", "--head_pose_path", required=True, type=str, 
		help="(required) Path to Head Pose Estimation Model .xml file.")
	## parser of gaze detection path
	parser.add_argument("-ge", "--gaze_estimation_path", required=True, type=str, 
		help="(required) Path to Gaze Estimation Model .xml file.")
	## parser of input path
	parser.add_argument("-i", "--input", required=True, type=str, 
		help="(required) Path to Input File either image or video or CAM (using camera).")
	## parser of display perf_counts on each model
	parser.add_argument("-pc", "--perf_counts",required=False, help="Report performance counters"
						"Print the real time takes for each layer in the model", 
						default=False, action="store_true")
	## parser of flags which choose way of visualization
	parser.add_argument("-show", "--flag_visualization", required=False, nargs='+', 
		default=[], help="(optional) Visualize the selected model on the output frame window. "
						 "'win': display the visualization window (To visualize other model must with 'win'), "
						 "'fd': Face Detection Model,			'fl': Facial Landmarks Detection Model, "
						 "'crop': Cropped face with Landmarks Detection,"
						 "'hp': Head Pose Estimation Model,	'ge': Gaze Estimation Model. "						 
						 "For example, '--show win fd fl hp ge crop' (Seperate each flag by space).")
	## parser of cpu_extension
	parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="(optional) MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to cpu_extension if layers from model are not supported on device.")
	## parser of device
	parser.add_argument("-d", "--device", required=False, type=str, default="CPU",
                        help="(optional) Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
	## parser of threshold
	parser.add_argument("-pt", "--prob_threshold", required=False, type=float, default=0.6,
                        help="(optional) Probability threshold for detections (0.6 by default)")

	return parser


if __name__=='__main__':
	main()	