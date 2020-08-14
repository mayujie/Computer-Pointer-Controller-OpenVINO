'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore

class HeadPoseEstimation:
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
        self.model_weights = self.model_name.split(".")[0] + '.bin'
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None


    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore()
        ## check if read model without problem
        self.check_model(self.model_structure, self.model_weights)        
        
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [ul for ul in self.network.layers.keys() if ul not in supported_layers]


        if len(unsupported_layers)!=0 and self.device=='CPU':
            print("unsupported layers found: {}".format(unsupported_layers))

            if not self.extensions==None:
                print("Adding cpu_extension now")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
                unsupported_layers = [ul for ul in self.network.layers.keys() if ul not in supported_layers]
                
                if len(unsupported_layers)!=0:
                    print("Please try again! unsupported layers found after adding the extensions")
                    exit(1)
                print("Problem is resolved after adding the extension!")
                
            else:
                print("Please give the right path of cpu extension!")
                exit(1)

        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)
        
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = [o for o in self.network.outputs.keys()]


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        # print(image.shape) # (374, 238, 3)
        processed_input = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:processed_input})
        last_output = self.preprocess_output(outputs)

        return last_output

    def check_model(self, model_structure, model_weights):
        # raise NotImplementedError
        try:
            # Reads a network from the IR files and creates an IENetwork, load IR files into their related class, architecture with XML and weights with binary file
            self.network = self.plugin.read_network(model=model_structure, weights=model_weights)
        except Exception as e:
            raise ValueError("Error occurred during head_pose_estimation network initialization.")

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # we wanna opposite order from H, W 
        # print(self.input_shape) # [1, 3, 60, 60]
        H, W = self.input_shape[2], self.input_shape[3]
        # print(H, W) # (60, 60)

        image_resized = cv2.resize(image, (W, H))
        # print(image_resized.shape) # (60, 60, 3)
        # (optional)
        # image_processed = np.transpose(np.expand_dims(image_resized, axis=0), (0, 3, 1, 2))
        
        # transpose so that order has channels 1st, cuz our image after resizing still have channels last
        # 1st put the 3rd channel which is our image channels for BGR. 
        # and next is 0 and 1 which were originally our heihgt and width of the image    
        image = image_resized.transpose((2,0,1))
        # print(image.shape) # (3, 60, 60)
        # add 1 dim at very start, then channels then H, W
        image_processed = image.reshape(1, 3, self.input_shape[2], self.input_shape[3])
        # print(image_processed.shape) # (1, 3, 60, 60)

        return image_processed

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
        Output layer names in Inference Engine format:

        name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
        name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
        name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).    

        z-axis is directed from person's eyes to the camera center
        y-axis is vertical, and x-axis is orthogonal to both z,y axes so that (x,y,z) constitute a right-handed coordinate system.
        '''
        # print(outputs) # dict of p, r, y
        # print(outputs.keys()) # dict_keys(['angle_p_fc', 'angle_r_fc', 'angle_y_fc'])
        ## [[6.9274316]] [[-4.026596]] [[-1.8397517]]
        # print(outputs['angle_y_fc'], outputs['angle_p_fc'], outputs['angle_r_fc'])
        outs_pry = []        
        for key in outputs:
            # print(key, outputs[key])
            outs_pry.append(outputs[key].tolist()[0][0])

        # print(outs_pry) # p, r, y        
        ## order is (2, 0, 1) # y, p, r
        ypr_outs = [outs_pry[i] for i in [2, 0, 1]]
        # print(ypr_outs) # [y, p, r]
        ## check order is correct
        # print('***', outputs['angle_y_fc'], outputs['angle_p_fc'], outputs['angle_r_fc'])

        return ypr_outs
