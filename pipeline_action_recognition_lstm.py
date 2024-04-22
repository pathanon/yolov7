# design network
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from tqdm import tqdm
from keras.models import Sequential
from keras.layers import LSTM

import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import pandas as pd
import time
import math
import argparse

from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
# from action_detection_yolov7 import get_action,get_distance

# Modified to make feature collector by Nontaphat
class PoseTimeCollector:
    def __init__(self):
        self.data = {}

    def add_value(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)
    
    def add_dict(self, dict_add):
        for keys in dict_add.keys():
            self.add_value(keys,dict_add[keys])

    def get_values(self, key):
        return self.data.get(key, [])
    
    def get_data(self):
        return self.data

def append_dicts(*dicts):
    appended_dict = {}
    for d in dicts:
        appended_dict.update(d)
    return appended_dict

def get_action(landmarks_list,landmarks_angle_list,diff):
    clap_angle_thres = 100.0
    walk_distance_thres = 10.0

    LE_angle = landmarks_angle_list['LE_angle']
    RE_angle = landmarks_angle_list['RE_angle']

    LWPts = landmarks_list["left_wrist"]
    RWPts = landmarks_list["right_wrist"]

    NPts = landmarks_list["nose"]
    if diff>=walk_distance_thres:
        return "walk"
    if LWPts[1]<NPts[1]:
        return "raise your left"
    else:
        if (RWPts is None):
            return "stand"
        if RWPts[1]<NPts[1]:
            return "raise your right"
        else:
            if (LE_angle<clap_angle_thres) and (RE_angle<clap_angle_thres):
                return "clap"
            else:
                return "stand"
    
            
def get_distance(lm1,lm2):
    return np.sqrt(np.sum((lm1-lm2)**2))

def prepare_keypoint_array(key_pts,angle_pts,xmin,xmax,ymin,ymax):
    features = []
    for key in key_pts.keys():
        fpts = key_pts[key]
        fpts[0]/=frame_width
        fpts[1]/=frame_height
        features.append(fpts)

    features.append([xmin/frame_width,xmax/frame_width])
    features.append([ymin/frame_height,ymax/frame_height])

    for key in angle_pts.keys():
        fpts = angle_pts[key]
        fpts *= (math.pi/360)
        features.append([fpts])

    return np.concatenate(features,axis=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='video_test.mp4', help='video input path')
    parser.add_argument('--output', type=str, default='result_lstm_video_test.mp4', help='video output path')

    parser.add_argument('--lstm', type=str, default='', help='lstm model path')
    parser.add_argument('--lstm_steps', type=int, default=5, help='number of steps that be used in input sequence (usually 5 frames collecting)')
    parser.add_argument('--collect', type=bool, default=False, help='set pipeline to collect pose features and action in each frame')
    parser.add_argument('--both', type=bool, default=False, help='set pipeline to show action both from LSTM and Rule-Based')
    opt = parser.parse_args()

    steps_shape = opt.lstm_steps
    features_shape = 40
    with tf.device('/cpu:0'): # to use GPU, change to "/cuda:0"
        if opt.lstm == '':
            print(f"......... use existing lstm .....steps....{steps_shape}....")
            model_prediction = tf.keras.models.load_model(f'./trained_lstm_model/model_steps_{steps_shape}/model')
        else:
            model_prediction = tf.keras.models.load_model(f'{opt.lstm}')
        print("-------- load model LSTM complete -----------")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load('yolov7-w6-pose.pt')
    print("-------- load model YOLOv7 complete -----------")
    model = weigths['model']
    model = model.half().to(device)
    _ = model.eval()
    if opt.collect:
        collector = PoseTimeCollector()

    # Load Video Input
    video_path = opt.input
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')
    
    # Get the frame width and height.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    # Pass the first frame through `letterbox` function to get the resized image,
    # to be used for `VideoWriter` dimensions. Resize by larger side.
    vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]
    
    save_name = f"{video_path.split('/')[-1].split('.')[0]}"
    # Define codec and create VideoWriter object .
    if opt.output:
        out = cv2.VideoWriter(f"{opt.output}",
                        cv2.VideoWriter_fourcc(*'mp4v'), 30,
                        (resize_width, resize_height))
    else:
        out = cv2.VideoWriter(f"{save_name}_action_lstm.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), 30,
                            (resize_width, resize_height))
    
    # initial value frame,total fps
    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    # initial the variable that will keep the state of leg in both left and right 
    old_legleft_pts = (frame_width//2,frame_height//2)
    old_legright_pts = (frame_width//2,frame_height//2)

    feature_collector_list = []

    while(cap.isOpened):
        # Capture each frame of the video.
        ret, frame = cap.read()
        if ret:
            # image preparing
            orig_image = frame
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            image = image.to(device)
            image = image.half()
            # Get the start time.
            start_time = time.time()
            with torch.no_grad():
                output, _ = model(image)
                # Get the end time.
            end_time = time.time()
            # Get the fps.
            fps = 1 / (end_time - start_time)
            # Add fps to total fps.
            total_fps += fps

            output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
            output = output_to_keypoint(output)

            nimg = image[0].permute(1, 2, 0) * 255
            nimg = nimg.cpu().numpy().astype(np.uint8)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
            if opt.collect:
                person_kpts_detect = dict()
                person_angles_detect = dict()
            #  to find the main person in the video
            person_idx = 0
            person_max_area = 0
            for idx in range(output.shape[0]):
                xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
                xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
                person_area = np.abs(xmax-xmin)*np.abs(ymax-ymin)
                if person_area > person_max_area:
                    person_max_area = person_area
                    person_idx = idx
            idx = person_idx # the index of main person in the video 

            kpts_detect, angles_detect = plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
            
            # get a current x,y position of leg on both left and right
            current_legleft_pts = kpts_detect['left_ankle']
            current_legright_pts = kpts_detect['right_ankle']
            # get difference between old and current x,y position of leg on both left and right
            differ_left = get_distance(old_legleft_pts,current_legleft_pts)
            differ_right = get_distance(old_legright_pts,current_legright_pts)
            maximum_differ = np.max([differ_left,differ_right])

            old_legleft_pts = current_legleft_pts.copy()
            old_legright_pts = current_legright_pts.copy()

            xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
            xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
        
            action = get_action(kpts_detect,angles_detect,maximum_differ)

            if opt.collect:
                person_kpts_detect[f"p{idx}"] = [kpts_detect,xmin,ymin,xmax,ymax]
                person_angles_detect[f"p{idx}"] = angles_detect
                collector.add_dict(kpts_detect)
                collector.add_dict(angles_detect)
                collector.add_value('Xbound',[xmin,xmax])
                collector.add_value('Ybound',[ymin,ymax])
                collector.add_value("frame_num",frame_count)
                collector.add_value("action",action)
            # LSTM

            new_kpts_detect = kpts_detect.copy()
            new_angles_detect = angles_detect.copy()
            features_points_collect = prepare_keypoint_array(new_kpts_detect,new_angles_detect,xmin,xmax,ymin,ymax)
            
            if frame_count<steps_shape:
                # collect pose first
                feature_collector_list.append(features_points_collect)
            else:
                # collect pose first
                f2t_arr = np.stack(feature_collector_list,axis=0)
                f2t_arr = np.array([f2t_arr])
                # print(f2t_arr.shape)
                res_pred = model_prediction.predict(f2t_arr)[0]
                # print(res_pred.shape)
                # print(f"{frame_count} do----------action {np.argmax(res_pred)}")
                feature_collector_list = feature_collector_list[1:]
                feature_collector_list.append(features_points_collect)
                # {'clap':0, 'raise your left':1, 'raise your right':2, 'stand':3, 'walk':4}
                action_list_name = ['clap','raise your left','raise your right','stand','walk']
                prediction_text_list = [
                    f"clap: {res_pred[0]*100:.2f} %",
                    f"raise your left: {res_pred[1]*100:.2f} %",
                    f"raise your right: {res_pred[2]*100:.2f} %",
                    f"stand: {res_pred[3]*100:.2f} %",
                    f"walk: {res_pred[4]*100:.2f} %",
                ]
                xstart,ystart = (int(xmax+10), int(ymin+50))
                steps_stride = 20
                font_name = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thinkness = 2
                for k,txt in enumerate(prediction_text_list):
                    cv2.putText(nimg,txt,(xstart,ystart+int(steps_stride * k)),font_name,font_scale,(255,90+int(30*k),90+int(30*k)),font_thinkness)
            # visualize

            cv2.rectangle(
                nimg,
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                color=(255, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA
            )
            if opt.both:
                font_name = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                pos = (int(xmin), int(ymin-10))
                action_text = f"action-rule: {action}"
                action_text_color=(255,120,120)

                cv2.putText(nimg, action_text, pos, font_name, font_scale, action_text_color, 2)
                
                if frame_count>steps_shape:
                    pos = (int(xmin), int(ymin-50))
                    action_text = f"action-lstm: {action_list_name[np.argmax(res_pred)]}"
                    action_text_color=(255,0,0)
                    cv2.putText(nimg, action_text, pos, font_name, font_scale, action_text_color, 2)


            if not opt.both:
                if frame_count>steps_shape:
                    pos = (int(xmin), int(ymin-10))
                    action_text = f"action: {action_list_name[np.argmax(res_pred)]}"
                    action_text_color=(255,0,0)
                    cv2.putText(nimg, action_text, pos, font_name, 1, action_text_color, 2)
            # Write the FPS on the current frame.
            cv2.putText(nimg, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            # Convert from BGR to RGB color format.
            cv2.imshow('image', nimg)
            out.write(nimg)
            # Increment frame count.
            frame_count += 1
            # Press `q` to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   
        else:
            break
    # Release VideoCapture().
    cap.release()
    # Close all frames and video windows.
    cv2.destroyAllWindows()
    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    if opt.collect:
        df = pd.DataFrame(collector.get_data())
        df.to_csv('collected_action_pose_frame.csv')