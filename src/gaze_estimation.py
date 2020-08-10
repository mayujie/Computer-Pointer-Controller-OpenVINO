'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore
import math

class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split(".")[0]+'.bin'
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore()
        self.network = self.plugin.read_network(model=self.model_structure, weights=self.model_weights)
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]


        if len(unsupported_layers)!=0 and self.device=='CPU':
            print("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions==None:
                print("Adding cpu_extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    print("After adding the extension still unsupported layers found")
                    exit(1)
                print("After adding the extension the issue is resolved")
            else:
                print("Give the path of cpu extension")
                exit(1)
                
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.network.outputs.keys()]

    def predict(self, left_eye_image, right_eye_image, hpa):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        leye_img_processed, reye_img_processed = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        outputs = self.exec_net.infer({'head_pose_angles':hpa, 'left_eye_image':leye_img_processed, 'right_eye_image':reye_img_processed})

        new_mouse_coord, gaze_vector = self.preprocess_output(outputs,hpa)

        return new_mouse_coord, gaze_vector


    def check_model(self):
        pass

    def preprocess_input(self, left_eye, right_eye):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # leye_image_resized = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
        # reye_image_resized = cv2.resize(right_eye, (self.input_shape[3], self.input_shape[2]))

        leye_image_resized = cv2.resize(left_eye, (60,60))
        reye_image_resized = cv2.resize(right_eye, (60,60))

        left_eye_final = leye_image_resized.transpose((2,0,1))
        right_eye_final = reye_image_resized.transpose((2,0,1))

        left_eye_final = left_eye_final.reshape(1, *left_eye_final.shape)
        right_eye_final = right_eye_final.reshape(1, *right_eye_final.shape)
        # (optional)
        # left_eye_final = np.transpose(np.expand_dims(leye_image_resized,axis=0), (0,3,1,2))
        # right_eye_final = np.transpose(np.expand_dims(reye_image_resized,axis=0), (0,3,1,2))
        
        return left_eye_final, right_eye_final


    def preprocess_output(self, outputs, hpa):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        (x, y, z) --> (y, p, r)
        r is gaze
        '''
        
        gaze_vector = outputs[self.output_names[0]].tolist()[0]
        #gaze_vector = gaze_vector / cv2.norm(gaze_vector)
        roll_value = hpa[2] #angle_r_fc output from HeadPoseEstimation model

        cos_value = math.cos(roll_value * math.pi / 180.0)
        sin_value = math.sin(roll_value * math.pi / 180.0)

        new_x = gaze_vector[0] * cos_value + gaze_vector[1] * sin_value
        new_y = gaze_vector[1] * cos_value - gaze_vector[0] *  sin_value

        return (new_x,new_y), gaze_vector