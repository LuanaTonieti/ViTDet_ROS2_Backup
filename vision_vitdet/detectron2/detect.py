#!/usr/bin/env python3
import numpy as np
import torch
import cv2
from telnetlib import NOP
import matplotlib.pyplot as plt

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from contextlib import ExitStack, contextmanager
from detectron2.engine.defaults import create_ddp_model

import glob
import time

import rclpy
from rclpy.node import Node
from custom_interfaces.msg import Vision

from matplotlib.animation import FuncAnimation

from .camvideostream import WebcamVideoStream


threshold = 0.35

cfg = LazyConfig.load("./src/vision_vitdet/vision_vitdet/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py")

# edit the config to utilize common Batch Norm
cfg.model.backbone.norm = "BN"
cfg.model.roi_heads.num_classes = 2

register_coco_instances("ball_robot", {},"./src/vision_vitdet/vision_vitdet/detectron2/Telstar_Mechta/train/_annotations.coco.json", "./src/vision_vitdet/vision_vitdet/detectron2/Telstar_Mechta/train")
MetadataCatalog.get("ball_robot").thing_classes = ['ball', 'robot']

# print(cfg)

cfg.train.device = "cpu"
model = instantiate(cfg.model)
model.to(cfg.train.device)
model = create_ddp_model(model)
DetectionCheckpointer(model).load("./src/vision_vitdet/vision_vitdet/detectron2/model_final_mask.pth")  # load a file, usually from cfg.MODEL.WEIGHTS


plt.ion()
plt.show()


class Detect(Node):

    def __init__(self):
        super().__init__('detect')
        self.get_logger().info('Running Vision VitDet Node')
        self.publisher_ = self.create_publisher(Vision, '/ball_position', 10)
        self.publisher_robot = self.create_publisher(Vision, '/robot_position', 10)
        self.ball = False
        self.vcap = WebcamVideoStream(src=1).start() # Abrindo camera
        self.timer=self.create_timer(0.008,self.timer_callback)

    def timer_callback(self):
        print("Inside detect()")
        msg_ball=Vision()
        self.ball = False
        start_timer = time.time()
        img = self.vcap.read()
        
        
        img2 = img
        img2 = torch.from_numpy(np.ascontiguousarray(img))
        img2 = img2.permute(2, 0, 1)  # HWC -> CHW
        #if torch.cuda.is_available():
        #    img = img.cuda()
        #    print("Available")
        #else:
        #    print("Running on CPU")
        inputs = [{"image": img2}]

        # run the model
        model.eval()
        with torch.no_grad():
            predictions_ls = model(inputs)
            # print(f'Time: {time.time() - start}')
        predictions = predictions_ls[0]

        indices = predictions['instances'].get_fields()['pred_classes'].to("cpu").numpy()
        classes = MetadataCatalog.get("ball_robot").thing_classes
        labels = [classes[idx] for idx in indices] 
        print(labels)

        print(predictions)

        for i in range(len(predictions["instances"])):
            #print(predictions["instances"][i])
            #print(predictions["instances"][0].get("pred_boxes"))
            score = predictions["instances"][i].get("scores").item()
            if (score>threshold):
                box = predictions["instances"][i].get("pred_boxes")
                box_tensor = box.tensor[0]
                x_inicial = box_tensor.data[0].item()
                y_inicial = box_tensor.data[1].item()
                x_final = box_tensor.data[2].item()
                y_final = box_tensor.data[3].item()
                start = (int(x_inicial), int(y_inicial))
                final = (int(x_final), int(y_final))
                cv2.rectangle(img, start, final, (255, 0, 0), 3)
                img = cv2.putText(img, labels[i], (int(x_inicial), int(y_inicial)-4), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 0, 0), 2, cv2.LINE_AA)
                img = cv2.putText(img, str(round(score,3)), (int(x_inicial), int(y_final)+14), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 0, 0), 2, cv2.LINE_AA)
                if labels[i]=='ball':
                    self.ball = True
                
        
        print(f'Time total: {time.time() - start_timer}')
        #cv2.imshow("RoboFEI",img)
        plt.clf()
        plt.rcParams["figure.figsize"] = [20, 20]
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=[0, 3500, 0, 2000])
        plt.show()
        plt.pause(0.001)
        
# as opencv loads in BGR format by default, we want to show it in RGB.
        
        if  0xFF == ord('q'):
            print("FINISHED SUCCESSFULLY!")
            cam.release()
            cv2.destroyWindow("RoboFEI")
        

        if self.ball==True:
            msg_ball.detected = True
            self.publisher_.publish(msg_ball)
        else:
            msg_ball.detected = False
            self.publisher_.publish(msg_ball)

        
        


def main(args=None):
    rclpy.init(args=args)
    
    detection = Detect()
    
    rclpy.spin(detection)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    detection.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



