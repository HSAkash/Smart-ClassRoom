import platform
import numpy as np
from glob import glob



# Known face embedding && which save as npy file
def get_known_face_embedding(path):
    """
    path: dirpath
    full path will be: {path}/*/*.npy
    """
    # emb_paths = glob(f"{path}/*/*.npy")
    emb_paths = glob(f"{path}/*/*.npy")
    if platform.system() == "Windows":
        known_face_ids = [x.split("\\")[-2] for x in emb_paths]
    else:
        known_face_ids = [x.split("/")[-2] for x in emb_paths]
    known_embedding = []
    for emb_path in emb_paths:
        known_embedding.append(np.load(emb_path))
    return np.array(known_embedding), np.array(known_face_ids)


# make recode for track
def get_records(detections, ids, all_keypoints):
    """
    detections: object of suppervision class
    ids: list of person id
    all_keypoints:all_keypoints: shape (n, 17, 3). Keypoints for n person.
    Here 0 point is nose. (x, y, confident)

    return: numpy array. shape (n, 3). n is number of person.
    (track_id, person_id, (x, y)) here (x, y) is nose point.
    """
    a = detections.tracker_id.copy()
    b = np.array(ids, dtype=object)
    c =  all_keypoints[:,0,:2].astype(np.int32).copy()
    a = np.reshape(a, (-1, 1))
    b = np.reshape(b, (-1, 1))
    return np.concatenate((a, b, c), axis=1)