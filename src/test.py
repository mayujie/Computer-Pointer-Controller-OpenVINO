import os
import cv2
import numpy as np
import time
import logging
from input_feeder import InputFeeder
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarkDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController

def main():

	File_path_input = "/Intel-AI/Computer-Pointer-Controller-OpenVINO/bin/demo.mp4"
	model_path = "/Intel-AI/Computer-Pointer-Controller-OpenVINO/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml"
	model_path = [
	"/Intel-AI/Computer-Pointer-Controller-OpenVINO/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml",
	"/Intel-AI/Computer-Pointer-Controller-OpenVINO/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml",
	"/Intel-AI/Computer-Pointer-Controller-OpenVINO/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml",
	"/Intel-AI/Computer-Pointer-Controller-OpenVINO/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml"
	]

	
	print('\n', File_path_input)
	print('\n', model_path)

	inputFeeder = InputFeeder("video", File_path_input)

	model_fd = FaceDetection(model_path[0], 'CPU')
	model_fld = FacialLandmarkDetection(model_path[1], 'CPU')
	model_hpe = HeadPoseEstimation(model_path[2], 'CPU')
	model_ge = GazeEstimation(model_path[3], 'CPU')


	inputFeeder.load_data()

	mouse_handler = MouseController('medium', 'fast')

	## load the model ##
	model_fd.load_model()
	model_fld.load_model()
	model_hpe.load_model()
	model_ge.load_model()

	frame_count = 0

	for ret, frame in inputFeeder.next_batch():
		# print(ret)

		if not ret:
			print('\n Video has reach to the end....')
			break

		event_key = cv2.waitKey(60)

		frame_count += 1

		# if frame_count % 1 == 0:
		# cv2.imshow('Video Stream', cv2.resize(frame, (500,500)))

		

		## Face detection ##
		cropped_face, face_coords = model_fd.predict(frame.copy(), 0.6)
		# print(cropped_face.shape)
		# print(face_coords)
		# print(type(cropped_face)
		## face_coords 
		## top left, bottom right

		## Landmarks detection ##
		l_eye_box, r_eye_box, eyes_coords = model_fld.predict(cropped_face.copy())
		# print(l_eye_box.shape, r_eye_box.shape) # left eye and right eye image
		## [left eye box, right eye box] 
		## [[leye_xmin,leye_ymin,leye_xmax,leye_ymax], [reye_xmin,reye_ymin,reye_xmax,reye_ymax]]
		# print(eyes_coords)
		
		## Head pose detection ##
		hpe_output = model_hpe.predict(cropped_face.copy())
		# [6.927431583404541, -4.0265960693359375, -1.8397517204284668]
		# print(hpe_output) # yaw, pitch, roll

		## Gaze estimation ##		
		mouse_position, gaze_vector = model_ge.predict(l_eye_box, r_eye_box, hpe_output)
		## mouse position (x, y), gaze_vector [-0.13984774, -0.38296703, -0.9055522 ]

		## Visualization the result
		frame_copy = frame.copy()
		if frame_count % 1 == 0:
			# Face
			cv2.rectangle(frame_copy, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 2) 
			# Facial Landmark left right eyes
			cv2.rectangle(frame_copy, (face_coords[0] + eyes_coords[0][0], face_coords[1] + eyes_coords[0][1]), (face_coords[0]+eyes_coords[0][2], face_coords[1]+eyes_coords[0][3]),(255,255,255), 2)
			cv2.rectangle(frame_copy, (face_coords[0] + eyes_coords[1][0], face_coords[1] + eyes_coords[1][1]), (face_coords[0]+eyes_coords[1][2], face_coords[1]+eyes_coords[1][3]),(255,255,255), 2)
			# Head Pose values
			cv2.putText(frame_copy, "Angles of Head Pose:", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
			cv2.putText(frame_copy, "Yaw: {:.1f}".format(hpe_output[0]), (10, 55), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
			cv2.putText(frame_copy, "Pitch: {:.1f}".format(hpe_output[1]), (10, 85), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
			cv2.putText(frame_copy, "Roll: {:.1f}".format(hpe_output[2]), (10, 115), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
			# Gaze arrow left right eyes
			x, y = gaze_vector[0:2]
			len_add = 400
			## eye left center point (x, y)
			eye_left_center = (int(face_coords[0] + (eyes_coords[0][0]+eyes_coords[0][2])/2), int(face_coords[1] + (eyes_coords[0][1]+eyes_coords[1][3])/2))
			## eye right center point (x, y)
			eye_right_center = (int(face_coords[0] + (eyes_coords[1][0]+eyes_coords[1][2])/2), int(face_coords[1] + (eyes_coords[1][1]+eyes_coords[1][3])/2))			
			
			cv2.arrowedLine(frame_copy, eye_left_center, (int(eye_left_center[0]+x*len_add), int(eye_left_center[1]-y*len_add)), (0,0,255), 3)
			cv2.arrowedLine(frame_copy, eye_right_center, (int(eye_right_center[0]+x*len_add), int(eye_right_center[1]-y*len_add)), (0,0,255), 3)

			cv2.imshow('Visualization', cv2.resize(frame_copy, (800,700)))

		
		if frame_count % 4 == 0:
			mouse_handler.move(mouse_position[0], mouse_position[1])	


		## only shown detected face ##
		# face_frame = frame.copy()
		# cv2.rectangle(face_frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 2) 
		# cv2.imshow('FaceDetection', cv2.resize(face_frame, (800,700)))
		## ------------------------------------------------------------ ##

		## cropped face with landmarks left and right eyes ##	
		land_frame = cropped_face.copy()
		cv2.rectangle(land_frame, (eyes_coords[0][0], eyes_coords[0][1]), (eyes_coords[0][2],eyes_coords[0][3]),(0,255,0), 1)
		cv2.rectangle(land_frame, (eyes_coords[1][0], eyes_coords[1][1]), (eyes_coords[1][2],eyes_coords[1][3]),(0,255,0), 1)
		cv2.imshow('FacialLandmark', cv2.resize(land_frame, (300, 400)))
		## ------------------------------------------------------------ ##
		
		## only show head pose value ##
		# headpose_frame = frame.copy()
		# cv2.putText(headpose_frame, "Angles of Head Pose:", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
		# cv2.putText(headpose_frame, "Yaw: {:.1f}".format(hpe_output[0]), (10, 55), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
		# cv2.putText(headpose_frame, "Pitch: {:.1f}".format(hpe_output[1]), (10, 85), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
		# cv2.putText(headpose_frame, "Roll: {:.1f}".format(hpe_output[2]), (10, 115), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
		# cv2.imshow('HeadPoseEstimation', cv2.resize(headpose_frame, (800,700)))				
		## ------------------------------------------------------------ ##

		## only show the gaze direction ##
		# gaze_frame = frame.copy() 
		# x, y = gaze_vector[0:2]
		# len_add = 400
		# ## eye left center point (x, y)
		# eye_left_center = (int(face_coords[0] + (eyes_coords[0][0]+eyes_coords[0][2])/2), int(face_coords[1] + (eyes_coords[0][1]+eyes_coords[1][3])/2))
		# ## eye right center point (x, y)
		# eye_right_center = (int(face_coords[0] + (eyes_coords[1][0]+eyes_coords[1][2])/2), int(face_coords[1] + (eyes_coords[1][1]+eyes_coords[1][3])/2))					
		# cv2.arrowedLine(gaze_frame, eye_left_center, (int(eye_left_center[0]+x*len_add), int(eye_left_center[1]-y*len_add)), (0,0,255), 3)
		# cv2.arrowedLine(gaze_frame, eye_right_center, (int(eye_right_center[0]+x*len_add), int(eye_right_center[1]-y*len_add)), (0,0,255), 3)
		# cv2.imshow('GazeEstimation', cv2.resize(gaze_frame, (800,700)))
		## ------------------------------------------------------------ ##

		if event_key ==27:
			break

	cv2.destroyAllWindows()
	inputFeeder.close()


if __name__ == '__main__':
	main()