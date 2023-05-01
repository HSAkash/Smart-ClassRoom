import face_recognition
import mediapipe as mp
import platform
from glob import glob
import cv2
import pandas as pd
import numpy as np

class PersonIdentityDetection:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
        self.mp_drawing = mp.solutions.drawing_utils

    def face_encoding(self, img):
        """
        img: numpy array
        """
        return face_recognition.face_encodings(img)[0]

    def get_knownFaceEncoding_label(self, path):
        """
        path: str, known images dir path
        """
        known_img_path = glob(f"{path}/*/*.jpg")


        if platform.system() == "Windows":
            self.known_face_ids = [x.split("\\")[-2] for x in known_img_path]
        else:
            self.known_face_ids = [x.split("/")[-2] for x in known_img_path]
        face_encodings = []
        for img_path in known_img_path:
            img = face_recognition.load_image_file(img_path)
            face_encodings.append(face_recognition.face_encodings(img)[0]) 

        self.known_face_encodings = np.array(face_encodings)
        return self.known_face_encodings, self.known_face_ids

    def is_face_inside(self, box, face_locations):
        b_x1, b_y1, b_x2, b_y2 = box
        f_y1, f_x2, f_y2, f_x1 = face_locations
        
        if b_x1 <= f_x1 and b_y1 <= f_y1 and b_x2 >= f_x2 and b_y2 >= f_y2:
            return True
            # rect1 is completely inside rect2
        else:
            # rect1 is not completely inside rect2
            return False
        
    def get_big_face_loaction(self, face_locations):
        if len(face_locations) == 0:
            return None
        elif len(face_locations) == 1:
            return face_locations[0]
        else:
            face_areas = [(x1 - x2) * (y1 - y2) for y1, x2, y2, x1 in face_locations]
            return face_locations[np.argmax(face_areas)]

    def get_percentance(self, box, face_locations):
        x1, y1, x2, y2 = box
        box_area = np.int64(x2-x1)*np.int64(y2-y1)
        y1, x2, y2, x1 = face_locations
        face_area = np.int64(x2-x1)*np.int64(y2-y1)
        return face_area/box_area

    def is_point_inside_rect(self, rect, points):
        count = 0
        for point in points:
            if rect[0]<=point[0]<=rect[2] and rect[1]<=point[1]<=rect[3]:
                count += 1
        if count / len(points) > .8:
            return True
        return False

    def pos_estimation_percentance(self, img, box, face_box, visibility=.9, threshold=.5):
        sample_img = img[box[1]:box[3],box[0]:box[2],  :]
        y1, x2, y2, x1 = face_box
        x1 -= box[0]
        x2 -= box[0]
        y1 -= box[1]
        y2 -= box[1]

        try:
            results = self.pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
        except:
            return False
        
        image_height, image_width, _ = sample_img.shape
        land_mark = []
        if results.pose_landmarks:

            # Iterate two times as we only want to display first two landmark.
            for i in range(len(results.pose_landmarks.landmark)):
                x = results.pose_landmarks.landmark[i].x * image_width
                y = results.pose_landmarks.landmark[i].y * image_height
                visibility = results.pose_landmarks.landmark[i].visibility
                land_mark.append((x,y, visibility))

        cols = ['x', 'y', 'visibility']
        df = pd.DataFrame(land_mark, columns=cols)
        df = df[(df['visibility']>visibility) & (df['x']<image_width) & (df['y'] < image_height)]
        minx = df.min().x
        maxx = df.max().x
        
        if abs(maxx - minx)/image_width < threshold:
            return False
        if not self.is_point_inside_rect((x1, y1, x2, y2), land_mark[:10]):
            return False
        
        return True
    
    def get_FacesLoaction(self, img):
        face_loaction = face_recognition.face_locations(img)
        return face_loaction

    def face_verification(self,img, face_loaction, known_embadings, known_face_ids, threshold=0.5):
        encoding = np.array(face_recognition.face_encodings(img, [face_loaction]))
        faceDis = face_recognition.face_distance(known_embadings, encoding)
        matchIndex = np.argmin(faceDis)
        Id = None
        if faceDis[matchIndex] < threshold:
            Id = known_face_ids[matchIndex]
        return Id 

    def get_identity(self, img, box, face_loactions, known_embadings, known_face_ids):
        all_faces = []
        face_inside = False
        big_face = None
        for face_loaction in face_loactions:
            if self.is_face_inside(box, face_loaction):
                all_faces.append(face_loaction)
        if len(all_faces) > 0:
            big_face = self.get_big_face_loaction(all_faces)
            box_percentance = self.get_percentance(box, big_face)

            if box_percentance > 0.15:
                face_inside = True
            else:
                face_inside = self.pos_estimation_percentance(img, box, big_face)
        Id = None
        if face_inside:
            Id = self.face_verification(img, big_face, known_embadings, known_face_ids)
        return Id

    def get_all_identities(self, img, detections, known_embadings, known_face_ids):
        face_loactions = self.get_FacesLoaction(img)
        Ids = []
        for detect in detections:
            box = detect[0]
            Id = self.get_identity(img, box, face_loactions, known_embadings, known_face_ids)
            Ids.append(Id)
        return Ids
