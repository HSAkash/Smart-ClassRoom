import platform
import numpy as np
import face_recognition



def get_face_locations(all_keypoints):
    """
    all_keypoints: shape (n, 17, 3). Keypoints for n person.
    Get from pose estimation. must be numpy array.
    Checking face location from keypoints.
    0-5 is face keypoints.
    """
    "___________________Find face location from keypoints______________________"
    
    
    "___________________Find face location from keypoints______________________"
    """
    y1, x2, y2, x1: shape (n,). n is number of person.
    where (y1, x1) is the top-left corner of the face bounding box and (y2, x2) is the bottom-right corner of the face bounding box
    and (y1, x1, y2, x2) is the face bounding box.
    Concatenate all face bounding box to face_locations. and Transpose to (4, n) shape.
    """
    face_locations = np.array([y1, x2, y2, x1], dtype=np.int32).T
    return face_locations


# check face confidence
def is_face_available(all_keypoints):
    """
    all_keypoints: shape (n, 17, 3). Keypoints for n person.
    Get from pose estimation. must be numpy array.
    Checking face location from keypoints.
    0-5 is face keypoints.
    0-3 is left eye, right eye, nose.
    if there is no face keypoints confidence > .9, return False.
    """
    mask = all_keypoints[:,:3,-1]>.9
    count_mask = np.sum(mask, axis=1)
    return count_mask == 3


# Face distance
def face_distance(face_encodings, face_to_compare):
    """
    face_encodings.shape = (n, 128)
    face_to_compare.shape = (k, 128)
    return_value = np.linalg.norm(face_encodings - face_to_compare, axis=1)
    return_value.shape = (k,)
    but i want to return shape = (n,k)
    """
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings[:, np.newaxis, :] - face_to_compare[np.newaxis, :, :], axis=-1)


def findCosineDistance(source_representation, test_representation):
    """
    source_representation.shape = (n,128)
    test_representation.shape = (k,128)
    return shape = (n,k)
    """
    a = np.matmul(source_representation, test_representation.T)
    b = np.sum(np.multiply(source_representation, source_representation), axis=1, keepdims=True)
    c = np.sum(np.multiply(test_representation, test_representation), axis=1, keepdims=True)
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


# Verify face
"""
thresholds:
    Dlib:
        cosine: 0.07
        linalg: 0.5
"""
def face_verification(
    known_face_embedding, test_embedding, known_face_ids, threshold=0.6, distance_type='linalg'):
    """
    thresholds:
        Dlib:
            cosine: 0.07
            linalg: 0.5
    """
    """
    known_face_embedding: face embedding of known person. which we extract from face image. and save as npy file.
    test_embedding: Run time person face embedding.
    known_face_ids: person id of known person label or identification.
    known_face_embedding.shape = (n,128)
    test_embedding.shape = (k,128)
    known_face_ids.shape = (n,)
    """
    known_face_ids = np.array(known_face_ids)
    if distance_type == "linalg":
        face_distances = face_distance(known_face_embedding, test_embedding)
    else:
        face_distances = findCosineDistance(known_face_embedding, test_embedding)
    
    # Find minimum distance
    faceDis = face_distances.min(axis=0)

    # Find minimum distance value index
    arg_min = face_distances.argmin(axis=0)

    # Find person id of minimum distance value
    ids = known_face_ids[arg_min]

    # Check if minimum distance value is less than threshold
    ids = np.where(faceDis<=threshold, ids, None)
    return ids



# Get person id from face embedding
def get_person_ids(
    frame, face_locations, face_available, boxes_conf,
    known_face_embedding, known_face_ids,
    trackId, recorder, threshold=0.5, distance_type='linalg'
):
    
    """
    frame: numpy array. shape (h, w, 3). h is height, w is width.(picture frame)
    face_locations: numpy array. shape (n, 4). n is number of person. 
    face_locations from keypoint estimation 0-5 keypoints.
    face_available: boolean numpy array. shape (n,). n is number of person.
    face_available from keypoint estimation 0-3 keypoints confident > .9.
    boxes_conf: numpy array. shape (n,). n is number of person.
    boxes_conf from person detection confident > .4.
    known_face_embedding: face embedding of known person. which we extract from face image. and save as npy file.
    known_face_ids: person id of known person label or identification.
    trackId: numpy array. shape (n,). n is number of person.
    trackId from Byte Tracker.
    recorder: object of TrackRecord class.
    """

    # Allready identified person id

    # Check trackId is in Identified_ids

    # Check person detection confident. If confident > .4, then True.

    """
    if trackId is not in Identified_ids and 
    that person face confident > .9 and person detection confident > .4 
    then we should identify that person.
    """
    is_face_check = []

    embeddings = []

    """___________________Face Embedding______________________"""
    """
    Those who are not in Identified_ids and are not face confident > .9 and person detection confident > .4
    Embedding those faces.
    """


    # get person id from face embedding varifications



    """___________________Labelling______________________"""
    lables = []

    


    """___________________duplicate value check____________________"""
    """
    if duplicate person id found, then remove old person id from Identified_ids.
    """


    """___________________duplicate value check____________________"""

    """___________________End labelling______________________"""

    return lables

