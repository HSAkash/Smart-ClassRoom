from yolox.tracker.byte_tracker import BYTETracker
from tracker import BYTETrackerArgs, detections2boxes, match_detections_with_tracks
from PersonIdentity import PersonIdentityDetection
from track_record import TrackRecoder
from utils_functions import get_records, update_records, get_labels
from tqdm import tqdm
from ultralytics import YOLO
import threading
import  supervision as sv 
import numpy as np


# MODEL
MODEL = "yolov8x.pt"
model = YOLO(MODEL)
model.fuse()

# Tracker
byte_tracker = BYTETracker(BYTETrackerArgs())


# Known face encoding
personIdentification = PersonIdentityDetection()
known_encodings, known_face_ids  = personIdentification.get_knownFaceEncoding_label("data/known_face")


# Record
recorder = TrackRecoder()
recorder.start_timer(60)

SOURCE_VIDEO_PATH = "data/input/kitiParty.mp4"
TARGET_VIDEO_PATH = "data/output/kitiParty.mp4"



generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# create instance of BoxAnnotator
annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=1)
# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# create instance of VideoWriter
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as writer:
    for frame in tqdm(generator, total=video_info.total_frames):
        # get boxes
        results = model(frame)[0]
        detections = sv.Detections.from_yolov8(results)
        detections = detections[detections.class_id == 0]
        new_tracker_list = [] 

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
        detections.xyxy = detections.xyxy.astype('int16')

        # nonidentify detect
        for detect in detections:
            if detect[3] not in recorder.Identified_ids:
                new_tracker_list.append(detect)

        
        if len(new_tracker_list) > 0:
            person_ids = personIdentification.get_all_identities(frame, new_tracker_list, known_encodings, known_face_ids)

        threading.Thread(target=update_records, args=(detections, new_tracker_list, person_ids, recorder,)).start()



        labels = get_labels(detections, recorder)
        # annotate boxes
        # frame = annotator.annotate(scene=frame, detections=detections, labels=labels)
        frame = annotator.annotate(scene=frame, detections=detections, labels=labels)


        # write frame
        writer.write_frame(frame)