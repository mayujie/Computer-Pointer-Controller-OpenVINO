'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore

class FaceDetectionModel:
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
        self.model_structure = self.model_name # model xml file
        self.model_weights = self.model_name.split('.')[0]+'.bin' # get model binary file path just use model xml file
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
        # load the IE Engine API plugin (Inference Engine entity)
        self.plugin = IECore()
        # Reads a network from the IR files and creates an IENetwork, load IR files into their related class, architecture with XML and weights with binary file
        self.network = self.plugin.read_network(model=self.model_structure, weights=self.model_weights)
        # Queries the plugin with specified device name what network layers are supported in the current configuration.
        # get the supported layers of the network
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        # check unsupported layer
        unsupported_layers = [ul for ul in self.network.layers.keys() if ul not in supported_layers]

        # condition of found unsupported layer and device is CPU
        if len(unsupported_layers)!=0 and self.device=='CPU':
            print('unsupported layers found:{}'.format(unsupported_layers))
            # extension is not None
            if not self.extensions==None:
                print("Adding cpu_extension")
                # Loads extension library to the plugin with a specified device name.
                self.plugin.add_extension(self.extensions, self.device)
                # update the support and unsupported layers
                supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
                unsupported_layers = [ul for ul in self.network.layers.keys() if ul not in supported_layers]
                # if still no unsupported layer exit
                if len(unsupported_layers)!=0:
                    print("After adding the extensions still unsupported layers found")
                    exit(1)
                print("After adding the extension the issue is resolved")
            # extensions is None exit    
            else:
                print("Give the path of cpu extension")
                exit(1)
        # Loads a network that was read from the Intermediate Representation (IR) to the plugin with specified device
        # load the network into the inference engine
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)
        
        # Get the input layer, iterate through the inputs here
        self.input_name = next(iter(self.network.inputs))
        # Return the shape of the input layer
        self.input_shape = self.network.inputs[self.input_name].shape
        # Get the output layer
        self.output_names = next(iter(self.network.outputs))
        # Return the shape of the output layer
        self.output_shape = self.network.outputs[self.output_names].shape


    def predict(self, image, prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        # 1.process the image
        img_processed = self.preprocess_input(image.copy())
        # 2.Starts synchronous inference for the first infer request of the executable network and returns output data.
        # A dictionary that maps output layer names
        outputs = self.exec_net.infer({self.input_name:img_processed})
        # print(outputs)

        # 3. process the outputs
        coords = self.preprocess_output(outputs, prob_threshold)
        # if coords empty, return 0,0
        if (len(coords)==0):
            return 0, 0
         # get the first detected face
        coords = coords[0]
        h=image.shape[0]
        w=image.shape[1]
        # print(coords)

        coords = coords* np.array([w, h, w, h])
        # Copy of the array, cast to a specified type. int32
        coords = coords.astype(np.int32)
        # print(coords)

        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]
        # print(cropped_face.shape)

        return cropped_face, coords


    def check_model(self):
        # raise NotImplementedError
        pass

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        Given an input image, height and width:
        '''
        # - Resize to height and width, (H, W), but resize use W, H which is opposite order
        # print(self.input_shape)
        image_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        # print(image_resized)
        # - Transpose the final "channel" dimension to be first to BGR
        # - Reshape the image to add a "batch" of 1 at the start
        img_processed = np.transpose(np.expand_dims(image_resized, axis=0), (0,3,1,2))
        # print(img_processed) # BxCxHxW

        return img_processed


    def preprocess_output(self, outputs, prob_threshold):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords = []
        outs = outputs[self.output_names][0][0] # output 
        for out in outs:
            conf = out[2]
            if conf > prob_threshold:
                x_min=out[3]
                y_min=out[4]
                x_max=out[5]
                y_max=out[6]
                coords.append([x_min, y_min, x_max, y_max])
        return coords