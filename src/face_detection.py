'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore

class FaceDetection:
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
        ## load the IE Engine API plugin (Inference Engine entity)
        self.plugin = IECore()
        ## check if read model without problem
        self.check_model(self.model_structure, self.model_weights)        
        self.exec_net = None
        ## Get the input layer, iterate through the inputs here
        self.input_name = next(iter(self.network.inputs))
        ## Return the shape of the input layer
        self.input_shape = self.network.inputs[self.input_name].shape
        ## Get the output layer
        self.output_names = next(iter(self.network.outputs))
        ## Return the shape of the output layer
        self.output_shape = self.network.outputs[self.output_names].shape
        

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        ## Queries the plugin with specified device name what network layers are supported in the current configuration.
        ## get the supported layers of the network
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        ## check unsupported layer
        unsupported_layers = [ul for ul in self.network.layers.keys() if ul not in supported_layers]

        ## condition of found unsupported layer and device is CPU
        if len(unsupported_layers)!=0 and self.device=='CPU':
            print('unsupported layers found: {}'.format(unsupported_layers))
            ## extension is not None
            if not self.extensions==None:
                print("Adding cpu_extension now")
                ## Loads extension library to the plugin with a specified device name.
                self.plugin.add_extension(self.extensions, self.device)
                ## update the support and unsupported layers
                supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
                unsupported_layers = [ul for ul in self.network.layers.keys() if ul not in supported_layers]
                ## if still no unsupported layer exit
                if len(unsupported_layers)!=0:
                    print("Please try again! unsupported layers found after adding the extensions.  device {}:\n{}".format(self.device, ', '.join(unsupported_layers)))
                    print("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
                    exit(1)
                print("Problem is resolved after adding the extension!")
            ## extensions is None exit    
            else:
                print("Please give the right path of cpu extension!")
                exit(1)
        ## Loads a network that was read from the Intermediate Representation (IR) to the plugin with specified device
        ## load the network into the inference engine
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)
        

    def predict(self, image, prob_threshold, perf_flag):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        ## 1.process the image
        processed_input = self.preprocess_input(image.copy())
        ## 2.Starts synchronous inference for the first infer request of the executable network and returns output data.
        ## A dictionary that maps output layer names
        outputs = self.exec_net.infer({self.input_name:processed_input})
        # print(outputs)
        
        if perf_flag:
            self.performance()

        ## 3. process the outputs
        coords = self.preprocess_output(outputs, prob_threshold)
        ## if coords empty, return 0,0
        if (len(coords)==0):
            return 0, 0
         ## get the first detected face
        coords = coords[0]
        h, w=image.shape[0], image.shape[1]
        ## print(coords, image.shape)        

        coords = coords* np.array([w, h, w, h])
        ## Copy of the array, cast to a specified type. int32
        coords = coords.astype(np.int32)
        ## (x_min, y_min) - coordinates of the top left bounding box corner
        ## (x_max, y_max) - coordinates of the bottom right bounding box corner.
        # print('top left, bottom right', coords)
        ## ymin:ymax, xmin:xmax --> height, width
        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]
        # print(cropped_face.shape)

        # cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (255,0,0), 2) 
        # cv2.imshow('detected face', cv2.resize(image, (600, 500)))

        return cropped_face, coords


    def check_model(self, model_structure, model_weights):
        # raise NotImplementedError
        try:
            # Reads a network from the IR files and creates an IENetwork, load IR files into their related class, architecture with XML and weights with binary file
            self.network = self.plugin.read_network(model=model_structure, weights=model_weights)
        except Exception as e:
            raise ValueError("Error occurred during face_detection network initialization.")


    def performance(self):
        perf_counts = self.exec_net.requests[0].get_perf_counts()
        # print('\n', perf_counts)
        print("## Face detection model performance:")
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))

        for layer, stats in perf_counts.items():            
            print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'], 
                                                              stats['status'], stats['real_time']))

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        Given an input image, height and width:
        '''
        ## - Resize to height and width, (H, W), but resize use W, H which is opposite order
        # print(image.shape)
        # print(self.input_shape) # [1, 3, 384, 672]
        H, W = self.input_shape[2], self.input_shape[3]
        # print(H, W) # (384, 672)

        image_resized = cv2.resize(image, (W, H))
        # print(image_resized.shape) # (384, 672, 3)
        ## - Transpose the final "channel" dimension to be first to BGR
        ## - Reshape the image to add a "batch" of 1 at the start
        ## (optional)
        # image_processed = np.transpose(np.expand_dims(image_resized, axis=0), (0,3,1,2))
        ## BxCxHxW
        image = image_resized.transpose((2,0,1))
        # print(image.shape) # (3, 384, 672)
        ## add 1 dim at very start, then channels then H, W
        image_processed = image.reshape(1, 3, self.input_shape[2], self.input_shape[3])
        # print(image_processed.shape) # (1, 3, 384, 672)

        return image_processed


    def preprocess_output(self, outputs, prob_threshold):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        coords = []
        # print(self.input_name)
        # print(self.output_names)
        # print(outputs[self.output_names].shape) # (1, 1, 200, 7)
        # print(outputs[self.output_names][0][0])
        outs = outputs[self.output_names][0][0] # output 
        for out in outs:
            # print(out)
            conf = out[2]
            if conf > prob_threshold:
                x_min=out[3]
                y_min=out[4]
                x_max=out[5]
                y_max=out[6]
                coords.append([x_min, y_min, x_max, y_max])
        return coords
