import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
import face_recognition


# video to frames extract and save
def video_to_frames(video_directory):
    """
    video_directory: Where video_directory is the path to the directory containing the video files.
    """
    video_paths = glob(f'{video_directory}/*.mp4')
    for video_path in tqdm(video_paths):
        video_name = os.path.basename(video_path)
        video_name = os.path.splitext(video_name)[0]
        os.makedirs(f'{video_directory}/{video_name}', exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i % 10 == 0:
                continue
            cv2.imwrite(f'{video_directory}/{video_name}/{i}.jpg', frame)
        cap.release()


# face embedding and save to npy
def face_embedding(image_directory, save_directory):
    """
    image_directory: Where image_directory is the path to the directory containing the image files directory.
    abs_path of image file is image_directory + individual_person_id  + image file name
    save_directory: save_directory is the path to the directory containing the save files (.npy) of face feature vactor.
    """
    image_paths = glob(f'{image_directory}/*/*.jpg')
    for image_path in tqdm(image_paths):
        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0]
        os.mkdir(f'{save_directory}/{image_name}', exist_ok=True)
        image_name = image_name.split('_')
        person_id = image_name[0]
        image_name = '_'.join(image_name[1:])
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            continue
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_encodings = np.array(face_encodings)
        np.save(f'{save_directory}/{person_id}_{image_name}.npy', face_encodings)
