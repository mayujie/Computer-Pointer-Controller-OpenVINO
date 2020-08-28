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
        self.plugin = IECore()
        ## check if read model without problem
        self.check_model(self.model_structure, self.model_weights)
        self.exec_net = None
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [o for o in self.network.outputs.keys()]
        

## check supported layer and performence counts reference: 
# https://gist.github.com/justinshenk/9917891c0433f33967f6e8cd8fcaa49a
    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        layers_unsupported = [l for l in self.network.layers.keys() if l not in supported_layers]


        if len(layers_unsupported)!=0 and self.device=='CPU':
            print("unsupported layers found: {}".format(layers_unsupported))

            if self.extensions!=None:
                print("Adding cpu_extension now")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                layers_unsupported = [l for l in self.network.layers.keys() if l not in supported_layers]
                
                if len(layers_unsupported)!=0:
                    print("Please try again! unsupported layers found after adding the extensions.  device {}:\n{}".format(self.device, ', '.join(layers_unsupported)))
                    print("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
                    exit(1)
                print("Problem is resolved after adding the extension!")
                
            else:
                print("Please give the right path of cpu extension!")
                exit(1)
                
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        

    def predict(self, left_eye_image, right_eye_image, hpa, perf_flag):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        ## (1, 3, 60, 60), (1, 3, 60, 60)
        leye_img_processed, reye_img_processed = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        # print('hpa=> ',hpa) ## yaw, pitch, roll
        outputs = self.exec_net.infer({'head_pose_angles':hpa, 'left_eye_image':leye_img_processed, 'right_eye_image':reye_img_processed})
        # print(outputs) # 'gaze_vector': array([[-0.13984774, -0.38296703, -0.9055522 ]]     
        
        if perf_flag:
            self.performance()

        new_mouse_coord, gaze_vector = self.preprocess_output(outputs,hpa)

        return new_mouse_coord, gaze_vector


    def check_model(self, model_structure, model_weights):
        # raise NotImplementedError
        try:
            # Reads a network from the IR files and creates an IENetwork, load IR files into their related class, architecture with XML and weights with binary file
            self.network = self.plugin.read_network(model=model_structure, weights=model_weights)
        except Exception as e:
            raise ValueError("Error occurred during gaze_estimation network initialization.")


## check supported layer and performence counts reference: 
# https://gist.github.com/justinshenk/9917891c0433f33967f6e8cd8fcaa49a
    def performance(self):
        perf_counts = self.exec_net.requests[0].get_perf_counts()
        # print('\n', perf_counts)
        print("\n## Gaze estimation model performance:")
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))

        for layer, stats in perf_counts.items():            
            print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'], 
                                                              stats['status'], stats['real_time']))


    def preprocess_input(self, left_eye, right_eye):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # leye_image_resized = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
        # reye_image_resized = cv2.resize(right_eye, (self.input_shape[3], self.input_shape[2]))
        # print(self.input_shape) # [1, 3, 60, 60]
        H, W = self.input_shape[2], self.input_shape[3]
        # print(H, W) # [60, 60]

        leye_image_resized = cv2.resize(left_eye, (W,H))
        reye_image_resized = cv2.resize(right_eye, (W,H))
        ## left (60, 60, 3) right (60, 60, 3)
        # print(leye_image_resized.shape, reye_image_resized.shape)

        trans_left_eye = leye_image_resized.transpose((2,0,1))
        trans_right_eye = reye_image_resized.transpose((2,0,1))
        ## left_trans (3, 60, 60) right_trans (3, 60, 60)
        # print(trans_left_eye.shape, trans_right_eye.shape)
        
        # print(*trans_left_eye.shape)
        left_eye_final = trans_left_eye.reshape(1, *trans_left_eye.shape)
        right_eye_final = trans_right_eye.reshape(1, *trans_right_eye.shape)
        ## left_eye_final (1, 3, 60, 60) right_eye_final (1, 3, 60, 60)
        # print(left_eye_final.shape, right_eye_final.shape)
                    
        ## (optional) ##
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
        # print(hpa) # [y, p, r]=[6.927431583404541, -4.0265960693359375, -1.8397517204284668]
        # print(outputs) # 'gaze_vector': array([[-0.13984774, -0.38296703, -0.9055522 ]]        
        gaze_vector = outputs[self.output_names[0]][0]
        # print(outputs[self.output_names[0]][0][1])
        # print(outputs[self.output_names[0]].tolist()[0])
        # print(gaze_vector) # [-0.13984774, -0.38296703, -0.9055522 ]
        #gaze_vector = gaze_vector / cv2.norm(gaze_vector)
        ## take angle_r_fc output from Head Pose Estimation
        angle_roll = hpa[2] 
        # Degree to Radian: (pi/180.0 * Deg)
        cos_val = math.cos(angle_roll * math.pi / 180.0)
        sin_val = math.sin(angle_roll * math.pi / 180.0)
        # print(cos_val, sin_val)

        # because we are draw mouse on 2D graph so need x, y
        x_val = gaze_vector[0] * cos_val + gaze_vector[1] * sin_val
        y_val = gaze_vector[1] * cos_val - gaze_vector[0] * sin_val
        # print('mouse coord: ', x_val, y_val)

        return (x_val, y_val), gaze_vector