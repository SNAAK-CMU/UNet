#---------------------------------------------------#
# Integrate single picture prediction, camera detection and FPS testing functions
# Integrated into a py file, the mode can be modified by specifying mode.
#---------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from unet import Unet_ONNX, Unet

if __name__ == "__main__":
    #--------------------------------------------------------------------------#
    # If you want to modify the corresponding type of color, just modify self.colors in the __init__ function.
    #--------------------------------------------------------------------------#
    #---------------------------------------------------------------------------------------------------------#
    # mode is used to specify the test mode:
    # 'predict' means single picture prediction. If you want to modify the prediction process, such as saving pictures, intercepting objects, etc., you can first read the detailed comments below.
    # 'video' means video detection, which can call the camera or video for detection. Please see the comments below for details.
    # 'fps' means testing fps. The picture used is street.jpg in img. For details, please see the comments below.
    # 'dir_predict' means traversing the folder to detect and save. By default, the img folder is traversed and the img_out folder is saved. See the comments below for details.
    # 'export_onnx' means exporting the model to onnx, which requires pytorch1.7.1 or above.
    # 'predict_onnx' means using the exported onnx model for prediction. The modification of relevant parameters is in Unet_ONNX around line unet.py_346
    #---------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #--------------------------------------------------------------------------#
    #count specifies whether to perform target pixel count (i.e. area) and proportion calculation
    # name_classes distinguishes types, the same as in json_to_dataset, used to print types and quantities
    #
    # count, name_classes are only valid when mode='predict'
    #--------------------------------------------------------------------------#
    count           = False
    name_classes    = ["background","inner_piece","outer_piece"]
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          Used to specify the path of the video. When video path = 0 means detecting the camera
    #                       If you want to detect videos, set the video path like = "xxx.mp4" That’s it, it means reading xxx in the root directory.MP4 files.
    #   video_save_path     Indicates the path to save the video. When video save path = "" means not to save
    #                       If you want to save the video, set the video save path like = "yyy.mp4"That’s it, it means saving it as yyy in the root directory..MP4 files.
    #   video_fps           fps used for saved video
    #
    #   Video path, video save path and video fps are only in mode = 'video'
    #   When saving the video, you need to ctrl+c to exit or run to the last frame to complete the complete saving step.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       Used to specify the number of image detections when measuring fps. Theoretically, the larger the test interval, the more accurate the fps.
    #   fps_image_path      fps image for specified test
    #   
    #   Test interval and fps image path are only in mode = 'fps'
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     Specifies the folder path to the images used for detection
    #   dir_save_path       Specifies the saving path of the detected images
    #   
    #   Dir origin path and dir save path are only valid when mode = 'dir predict'
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   simplify            Using Simplified onnx
    #   onnx_save_path      Specifies the saving path of onnx
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode != "predict_onnx":
        unet = Unet()
    else:
        yolo = Unet_ONNX()

    if mode == "predict":
        '''
        There are several points to note about predict.py
        1. This code cannot directly perform batch prediction. If you want to perform batch prediction, you can use os.listdir() to traverse the folder and use Image.open to open the image file for prediction.
        For the specific process, please refer to get_miou_prediction.py. The traversal is implemented in get_miou_prediction.py.
        2. If you want to save, use r_image.save("img.jpg") to save.
        3. If you want the original image and the segmented image not to be mixed, you can set the blend parameter to False.
        4. If you want to obtain the corresponding area based on the mask, you can refer to the detect_image function, which uses the prediction result drawing part to determine the type of each pixel, and then obtain the corresponding part according to the type.
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                output = input("output image filename: ")
                r_image.show()
                cv2.imwrite(output, np.array(r_image))
                

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("The camera (video) cannot be read correctly. Please pay attention to whether the camera is installed correctly (whether the video path is filled in correctly).")

        fps = 0.0
        while(True):
            t1 = time.time()
            # Read a frame
            ref, frame = capture.read()
            if not ref:
                break
            # Format conversion, bgr to rgb
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Convert to image
            frame = Image.fromarray(np.uint8(frame))
            # Perform testing
            frame = np.array(unet.detect_image(frame))
            # Rg bto bgr meets opencv display format
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = unet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        unet.convert_to_onnx(simplify, onnx_save_path)
                
    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
