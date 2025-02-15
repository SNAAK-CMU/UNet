import colorsys
import copy
import time
import os


import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


# ------------------------------------------------#
# Using your own trained model prediction requires modifying 2 parameters.
# Both model_path and num_classes need to be modified!
# If there is a shape mismatch
# Be sure to pay attention to the modification of model_path and num_classes during training.
# ------------------------------------------------#
class Unet(object):
    _defaults = {
        # ------------------------------------------------------------------#
        # model_path points to the weight file in the logs folder
        # After training, there are multiple weight files in the logs folder. Just select the one with the lower loss in the verification set.
        # A lower loss on the verification set does not mean a higher miou, it only means that the weight has better generalization performance on the verification set.
        # ------------------------------------------------------------------#
        "model_path": "logs/cheese/top_and_other/best_epoch_weights.pth",
        # --------------------------------#
        # The number of classes that need to be distinguished +1 (for background)
        # --------------------------------#
        "num_classes": 3,
        # --------------------------------#
        # Backbone network to be used. Options: "vgg", "resnet50"
        # --------------------------------#
        "backbone": "vgg",
        # --------------------------------#
        # Enter the size of the image
        # --------------------------------#
        "input_shape": [640, 640],
        # ------------------------------------------------#
        # The mix_type parameter is used to control the way the detection results are visualized.
        #
        # When mix_type = 0, it means that the original image and the generated image are mixed.
        # mix_type = 1 means only retaining the masks
        # When mix_type = 2, it means that only the background is deducted and only the target in the original image is retained.
        # ------------------------------------------------#
        "mix_type": 1,
        # --------------------------------#
        # Whether to use Cuda
        # Set to False if there is no GPU available
        # --------------------------------#
        "cuda": True,
    }

    # --------------------------------------------------#
    # Initialize UNET
    # --------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # --------------------------------------------------#
        # Set different colors for the picture frame
        # --------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (250, 250, 55), (250, 106, 77), (0, 0, 128), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # --------------------------------------------------#
        # Create the model
        # --------------------------------------------------#
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   Set up the netwoek and device
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        self.net = unet(num_classes = self.num_classes, backbone=self.backbone)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+"/"+self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   Segment the image
    # ---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        # ---------------------------------------------------------#
        # Convert the image to an RGB image here to prevent grayscale images from causing errors when predicting.
        # The code only supports prediction of RGB images, all other types of images will be converted to RGB
        # ---------------------------------------------------------#
        image       = cvtColor(image)
        # ---------------------------------------------------#
        #   Make a backup of the input image for later use in visualizing results
        # ---------------------------------------------------#
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   Add gray bars to the image to achieve distortion-free resize
        #   You can also directly resize for identification.
        # ---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        # ---------------------------------------------------------#
        #   Add the batch size dimension
        # ---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   The image is passed into the network for prediction
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   Get the type of each pixel
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            # --------------------------------------#
            #   Cut off the gray bar part
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            #   Resize images
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   Get the type of each pixel
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        # ---------------------------------------------------------#
        #   count
        # ---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            # ------------------------------------------------#
            #   Convert new picture to image form
            # ------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            # ------------------------------------------------#
            #   Mix the new image with the original image
            # ------------------------------------------------#
            image   = Image.blend(old_img, image, 0.3)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            # ------------------------------------------------#
            #   Convert new picture to image form
            # ------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            # ------------------------------------------------#
            #   Convert new picture to image form
            # ------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))

        return image

    def get_FPS(self, image, test_interval):
        # ---------------------------------------------------------#
        #   Here, the image is converted into an RGB image to prevent grayscale images from causing errors during prediction.
        #   The code only supports prediction of RGB images, all other types of images will be converted to RGB
        # ---------------------------------------------------------#
        image       = cvtColor(image)
        # ---------------------------------------------------------#
        #   Add gray bars to the image to achieve distortion-free resize
        #   You can also directly resize for identification.
        # ---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        # ---------------------------------------------------------#
        #   Add the batch size dimension
        # ---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   The image is passed into the network for prediction
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   Get the type of each pixel
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            # --------------------------------------#
            #   Cut off the gray bar part
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------#
                #   The image is passed into the network for prediction
                # ---------------------------------------------------#
                pr = self.net(images)[0]
                # ---------------------------------------------------#
                #   Get the type of each pixel
                # ---------------------------------------------------#
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                # --------------------------------------#
                #   Cut off the gray bar part
                # --------------------------------------#
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        import onnxsim
        import onnxruntime

        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:

            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_miou_png(self, image):
        # ---------------------------------------------------------#
        #   Here, the image is converted into an RGB image to prevent grayscale images from causing errors during prediction.
        #   The code only supports prediction of RGB images, all other types of images will be converted to RGB
        # ---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   Add gray bars to the image to achieve distortion-free resize
        #   You can also directly resize for identification.
        # ---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        # ---------------------------------------------------------#
        #   Add the batch size dimension
        # ---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   The image is passed into the network for prediction
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   Get the type of each pixel
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            # --------------------------------------#
            #   Cut off the gray bar part
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            #   Resize images
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   Get the type of each pixel
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image

class Unet_ONNX(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   Onnx path points to the onnx weight file in the model data folder
        #-------------------------------------------------------------------#
        "onnx_path"    : 'model_data/models.onnx',
        #--------------------------------#
        #   The number of classes that need to be distinguished +1 (for background)
        #--------------------------------#
        "num_classes"   : 21,
        #--------------------------------#
        #   Backbone network to be used. Options: "vgg", "resnet50"   
        #--------------------------------#
        "backbone"      : "vgg",
        #--------------------------------#
        #   Enter the image size
        #--------------------------------#
        "input_shape"   : [512, 512],
        #-------------------------------------------------#
        #   The mix_type parameter is used to control how the detection results are visualized.
        #
        #   When mix_type = 0, it means that the original image and the generated image are mixed.
        #   mix_type = 1 means only retaining the generated graph
        #   When mix_type = 2, it means that only the background is deducted and only the target in the original image is retained.
        #-------------------------------------------------#
        "mix_type"      : 0,
    }
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initialize model
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        
        self.onnx_session   = onnxruntime.InferenceSession(self.onnx_path)
        # Get all input nodes
        self.input_name     = self.get_input_name()
        # Get all output nodes
        self.output_name    = self.get_output_name()

        #---------------------------------------------------#
        #   Picture frame set different colors
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        show_config(**self._defaults)

    def get_input_name(self):
        # Get all input nodes
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_output_name(self):
        # Get all output nodes
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_feed(self,image_tensor):
        # Use input name to get the input tensor
        input_feed={}
        for name in self.input_name:
            input_feed[name]=image_tensor
        return input_feed
    
    #---------------------------------------------------#
    #   resize the input image
    #---------------------------------------------------#
    def resize_image(self, image, size):
        iw, ih  = image.size
        w, h    = size

        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))

        return new_image, nw, nh

    #---------------------------------------------------#
    #   Detect pictures
    #---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        #---------------------------------------------------------#
        #   Here, the image is converted into an RGB image to prevent grayscale images from causing errors during prediction.
        #   The code only supports prediction of RGB images, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   Make a backup of the input image for later use in drawing
        #---------------------------------------------------#
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   Add gray bars to the image to achieve distortion-free resize
        #   You can also directly resize for identification.
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   Add batch size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        input_feed  = self.get_input_feed(image_data)
        pr          = self.onnx_session.run(output_names=self.output_name, input_feed=input_feed)[0][0]

        def softmax(x, axis):
            x -= np.max(x, axis=axis, keepdims=True)
            f_x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
            return f_x
        print(np.shape(pr))
        #---------------------------------------------------#
        #   Get the type of each pixel
        #---------------------------------------------------#
        pr = softmax(np.transpose(pr, (1, 2, 0)), -1)
        #--------------------------------------#
        #   Cut off the gray bar part
        #--------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #---------------------------------------------------#
        #   Resize images
        #---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        #---------------------------------------------------#
        #   Get the type of each pixel
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)
        
        #---------------------------------------------------------#
        #   count
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   Convert new picture to image form
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            #   Mix the new image with the original image
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   Convert new picture to image form
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   Convert new picture to image form
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
        
        return image
