import cv2
import os
import logging
import numpy as np
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarkDetectionModel
from head_pose_estimation import HeadPoseEstimationModel
from gaze_estimation import GazeEstimationModel
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder
import time


def build_argparser():
	'''
	Parse command line arguments.
	return: command line arguments
	'''

	parser = ArgumentParser()
	parser.add_argument("-f", "--facedetectionmodel", required=True, type=str, 
						help="Specify Path to .xml file of Face Detection model.")
	parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str, 
						help="Specify Path to .xml file of Facial Landmark Detection model.")
	parser.add_argument("-hp", "--headposemodel", required="True", type=str , 
						help="Specify Path to .xml file of Head Pose Estimation model.")
	parser.add_argument("-g", "--gazeestimationmodel", required=True, type=str , 
						help="Specify Path to .xml file of Gaze Estimation model.")
	parser.add_argument("-i", "--input", required=True, type=str , 
						help="Specify Path to video file or enter cam for webcam")
	parser.add_argument("-flags", "--previewFlags", required=False, nargs='+', 
						default=[], 
						help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
	parser.add_argument("-l", "--cpu_extension", required=False, type=str , 
						default=None,
						help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
	parser.add_argument("-prob", "--prob_threshold", required=False, type=float, 
						default=0.6, 
						help="Probability threshold for model to detect the face accurately from the video frame.")
	parser.add_argument("-d", "--device", type=str, default="CPU", 
						help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
	
	return parser


def main():
	'''
	Grab command line args
	'''
	args = build_argparser().parse_args()
	previewFlags = args.previewFlags

	logging.basicConfig(filename='Project_log.log', level=logging.INFO)
	logger = logging.getLogger()
	inputFilePath = args.input
	inputFeeder = None
	if inputFilePath.lower()=="cam":
		inputFeeder = InputFeeder("cam")
	else:
		if not os.path.isfile(inputFilePath):
			logger.error("Unable to find specified video file")
			exit(1)
		inputFeeder = InputFeeder("video", inputFilePath)

	modelPathDict = {'FaceDetectionModel':args.facedetectionmodel, 'FacialLandmarkDetectionModel':args.faciallandmarkmodel,
	'HeadPoseEstimationModel':args.headposemodel, 'GazeEstimationModel':args.gazeestimationmodel}

	for fileNameKey in modelPathDict.keys():
		if not os.path.isfile(modelPathDict[fileNameKey]):
			logger.error("Unable to find specified "+fileNameKey+" xml file")
			exit(1)

	fdm = FaceDetectionModel(modelPathDict['FaceDetectionModel'], args.device, args.cpu_extension)
	fldm = FacialLandmarkDetectionModel(modelPathDict['FacialLandmarkDetectionModel'], args.device, args.cpu_extension)
	gem = GazeEstimationModel(modelPathDict['GazeEstimationModel'], args.device, args.cpu_extension)
	hpem = HeadPoseEstimationModel(modelPathDict['HeadPoseEstimationModel'], args.device, args.cpu_extension)

	mc = MouseController('medium', 'fast')

	inputFeeder.load_data()

	start_time = time.time()
	fdm.load_model()
	# fdm_load_time = time.time()
	fldm.load_model()
	# fldm_load_time = time.time()
	hpem.load_model()
	# hpem_load_time = time.time()
	gem.load_model()
	# gem_load_time = time.time()
	total_load_time = time.time() - start_time

	frame_count = 0
	start_infer_time = time.time()

	for ret, frame in inputFeeder.next_batch():
		if not ret:
			break
		frame_count+=1
		if frame_count %5 == 0:
			cv2.imshow('video', cv2.resize(frame, (900,700)))

		key = cv2.waitKey(60)
		# face detection model
		croppedFace, face_coords = fdm.predict(frame.copy(), args.prob_threshold)

		if type(croppedFace)==int:
			logger.error("Unable to detect the face.")
			# Esc key is 27
			if key==27:
				break
			continue

		# head pose estimation model
		hp_out = hpem.predict(croppedFace.copy())
		# facial landmarks detection model
		left_eye, right_eye, eye_coords = fldm.predict(croppedFace.copy())
		# gaze estimation model
		# needs input of left eye & right eye & head pose
		new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_out)

		if (not len(previewFlags)==0):
			preview_frame = frame.copy()

			if 'fd' in previewFlags:
				# cv2.rectangle(preview_frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 3) 
				preview_frame = croppedFace

			if 'fld' in previewFlags:
				cv2.rectangle(croppedFace, (eye_coords[0][0]-30, eye_coords[0][1]-30), (eye_coords[0][2]+30, eye_coords[0][3]+30), (255,255,255), 2)
				cv2.rectangle(croppedFace, (eye_coords[1][0]-30, eye_coords[1][1]-30), (eye_coords[1][2]+30, eye_coords[1][3]+30), (255,255,255), 2)
                # preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = croppedFace

			if 'hp' in previewFlags:
				cv2.putText(preview_frame, "Pose Angles:  yaw: {:.2f} | pitch: {:.2f} | roll: {:.2f}".format(hp_out[0],hp_out[1],hp_out[2]), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.25, (0, 255, 255), 1)
			
			if 'ge' in previewFlags:
				x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
				le =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
				cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
				re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
				cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
				croppedFace[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]] = le
				croppedFace[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]] = re
                # preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = croppedFace

			cv2.imshow("visualization",cv2.resize(preview_frame,(600,600)))

		if frame_count%5==0:
			mc.move(new_mouse_coord[0], new_mouse_coord[1])

		if key==27:
			break

	total_infer_time = time.time() - start_infer_time
	fps = frame_count / round(total_infer_time, 2)


	# logger.info('FaceDetectionModel load time: ' + str(fdm_load_time))
	# logger.info('FacialLandmarkDetectionModel load time: ' + str(fldm_load_time))
	# logger.info('HeadPoseEstimationModel load time: ' + str(hpem_load_time))
	# logger.info('GazeEstimationModel load time: ' + str(gem_load_time))
	logger.info('Model Load time: ' + str(total_load_time))
	logger.info('Inference time: ' + str(total_infer_time))
	logger.info('FPS: ' + str(fps))

	logger.error("VideoStream ended...")
	cv2.destroyAllWindows()
	inputFeeder.close()



if __name__ == '__main__':
	main()