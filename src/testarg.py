import os
from argparse import ArgumentParser
from input_feeder import InputFeeder

def build_argparser():

	parser = ArgumentParser()
	## parser of face detection path
	parser.add_argument("-fd", "--face_detection_path", required=True, type=str, 
		help="Path to Face Detection Model .xml file.")
	## parser of landmark detection path
	parser.add_argument("-fl", "--facial_landmarks_path", required=True, type=str, 
		help="Path to Facial Landmarks Detection Model .xml file.")
	## parser of headpose detection path
	parser.add_argument("-hp", "--head_pose_path", required=True, type=str, 
		help="Path to Head Pose Estimation Model .xml file.")
	## parser of gaze detection path
	parser.add_argument("-ge", "--gaze_estimation_path", required=True, type=str, 
		help="Path to Gaze Estimation Model .xml file.")
	## parser of input path
	parser.add_argument("-i", "--input", required=True, type=str, 
		help="Path to Input File either image or video or CAM (using camera).")
	## parser of flags which choose way of visualization
	parser.add_argument("-show", "--flag_visualization", required=False, nargs='+', 
		default=[], help="Visualize the selected model on the output frame. For example, '--show fd fld hp ge' (Seperate each flag by space)"
						 "fd: Face Detection Model,			fld: Facial Landmarks Detection Model"
						 "hp: Head Pose Estimation Model,	ge: Gaze Estimation Model")
	## parser of cpu_extension
	parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
	## parser of device
	parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
	## parser of threshold
	parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections (0.6 by default)")
	
	return parser


def main():
	args = build_argparser().parse_args()
	visual_flags = args.flag_visualization
	input_path = args.input

	Dict_model_path = {
		'Face': args.face_detection_path,
		'Landmarks': args.facial_landmarks_path,
		'Headpose': args.head_pose_path,
		'Gaze': args.gaze_estimation_path
	}

	if input_path == "CAM" or input_path=="cam":
		print("\n## You are using CAMERA right now..." + input_path.lower() + " detected!")
		input_feeder = InputFeeder(input_path.lower())		
	else:
		if not os.path.isfile(input_path):
			print("\n## Input file not exists in Path: " + input_path + ". Please check again !!!")
			exit(1)
		else:
			print('\n## Input path exists: '+ input_path + '\n')
			input_feeder = InputFeeder("video", input_path)

	for model_key in Dict_model_path.keys():
		print(Dict_model_path[model_key])
		if not os.path.isfile(Dict_model_path[model_key]):
			print("\n## " + model_key + " Model path not exists: " + Dict_model_path[model_key] + ' Please try again !!!')
			exit(1)
		else:
			print('## '+model_key + " Model path is correct: " + Dict_model_path[model_key])


	print(input_feeder)
	print(input_path)
	print(visual_flags)
	print(args.cpu_extension)
	print(args.device)
	print(args.prob_threshold)


if __name__=='__main__':
	main()