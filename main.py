import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import supervision as sv 
from ultralytics import YOLO
from track_record import TrackRecoder
from yolox.tracker.byte_tracker import BYTETracker
from utils_functions import get_records, get_known_face_embedding
from FaceRecognition import get_face_locations, is_face_available, get_person_ids
from tracker import BYTETrackerArgs, detections2boxes, match_detections_with_tracks



SOURCE_VIDEO_PATH = "dataset/video/input.mp4"
TARGET_VIDEO_PATH = "dataset/video/output.mp4"
INFO_FILE_PATH = "dataset/info.csv"
KNOWN_FACE_EMBEDDING_FILE_DIRECTORY_PATH = "dataset"


# Get person info for csv file.(Roll,Name)(Id,Name)
name_df = pd.read_csv(INFO_FILE_PATH)
name_df['Roll'] = name_df['Roll'].astype(str)
name_dict = name_df.set_index('Roll')['Name'].to_dict()


# Get known face encoding and label
known_face_embedding, known_face_ids  = get_known_face_embedding(KNOWN_FACE_EMBEDDING_FILE_DIRECTORY_PATH)



# MODEL
MODEL = "yolov8x-pose.pt"
model = YOLO(MODEL)

# Tracker
byte_tracker = BYTETracker(BYTETrackerArgs())


# Record
recorder = TrackRecoder()
# 5s timer for save record as csv file
recorder.start_timer(5)


# Generator which generate frame from video
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# create instance of BoxAnnotator
annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=1)
# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)


start_time = time.perf_counter ()
fps = 8
fps_count = 0


"""___________________________Attendance initialize_________________________________"""
"""
Take attendance
"""
# attendace_dict = {}
# attendace_count = 0

"""__________________________End Attendace initialize________________________________"""



# create instance of VideoWriter
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as writer:
    for frame in tqdm(generator, total=video_info.total_frames):
        # get boxes
        results = model(frame)[0]
        detections = sv.Detections.from_yolov8(results)
        if detections.class_id.shape[0]!=0:
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )

            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)



            # filtering out detections without trackers
            mask = np.array([tracker_id is not None  for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            detections.xyxy = detections.xyxy.astype('int32')


            boxes = results.boxes.xyxy.cpu().numpy()[mask==True]
            boxes_conf = results.boxes.conf.cpu().numpy()[mask==True]
            all_keypoints = results.keypoints.cpu().numpy()[mask==True]
            
            """___________________________Face Recognition_______________________________"""
            
            
            "____________________Draw____________________"
            # draw fps on frame
            fps_count += 1
            end_time = time.perf_counter ()
            if end_time - start_time >= 1.:
                start_time = time.perf_counter ()
                fps = fps_count
                fps_count = 0
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # frame = annotator.annotate(scene=frame, detections=detections, labels=ids)

            # draw labels on frame
            labels = [None]*len(ids)
            for i, id in enumerate(ids):
                if id:
                    labels[i] = f"{id} {name_dict[id]}"

            frame = annotator.annotate(scene=frame, detections=detections, labels=labels)
            "___________________End Draw_____________________"


            "___________________Start Take Attendance______________________"

            
            "___________________End Take Attendance________________________"



        # write frame
        writer.write_frame(frame)
recorder.stop_timer()