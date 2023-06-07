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
    all_min_x = all_keypoints[:,:5,0].min(axis=1)
    all_max_x = all_keypoints[:,:5,0].max(axis=1)
    all_mid_x = all_keypoints[:,0,0]

    all_min_y = all_keypoints[:,:5,1].min(axis=1)
    all_max_y = all_keypoints[:,:5,1].max(axis=1)
    all_mid_y = all_keypoints[:,0,1]
    distance = np.max(
        np.absolute(
            [
                all_mid_x - all_min_x,
                all_max_x - all_mid_x,
                all_mid_y - all_min_y,
                all_max_y - all_mid_y
            ]
        ), axis=0
    )

    y1 = all_mid_y - distance
    x2 = all_max_x
    y2 = all_mid_y + distance
    x1 = all_min_x

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
    Identified_ids = recorder.Identified_ids
    # Check trackId is in Identified_ids
    isin_trackId = np.isin(trackId, Identified_ids, invert=True)
    # Check person detection confident. If confident > .4, then True.
    is_boxes = boxes_conf > .4
    """
    if trackId is not in Identified_ids and 
    that person face confident > .9 and person detection confident > .4 
    then we should identify that person.
    """
    is_face_check = isin_trackId & face_available & is_boxes

    embeddings = []

    """___________________Face Embedding______________________"""
    """
    Those who are not in Identified_ids and are not face confident > .9 and person detection confident > .4
    Embedding those faces.
    """
    check_location = []
    for i in range(trackId.shape[0]):
        if is_face_check[i]:
            check_location.append(face_locations[i])
    check_location = np.array(check_location)
    embeddings = face_recognition.face_encodings(frame, check_location)
    """_________________________________________"""
    embeddings = np.array(embeddings)

    # get person id from face embedding varifications
    if embeddings.shape[0] > 0:
        ids = face_verification(known_face_embedding, embeddings, known_face_ids,
                            threshold=threshold, distance_type='linalg')


    """___________________Labelling______________________"""
    lables = []
    j = 0
    for i in range(face_locations.shape[0]):
        try:
            # Those who already identified get their id from track recoder Identified_ids_link_person_ids
            if not isin_trackId[i]:
                lables.append(recorder.Identified_ids_link_person_ids[trackId[i]])
            # Identify new person
            elif is_face_check[i]:
                lables.append(ids[j])
                j+=1
            # New person but not identified
            else:
                lables.append(None)
        except Exception as e:
            raise Exception(f"{e}")
    


    """___________________duplicate value check____________________"""
    """
    if duplicate person id found, then remove old person id from Identified_ids.
    """
    np_labels = np.array(lables)
    valid_indices = np.where(np_labels != None)[0]
    filtered_lables = np_labels[valid_indices]
    if len(filtered_lables)>0:
        if np.unique(filtered_lables, return_counts=True)[1].max() > 1:
            labels_dict = {}
            for i in range(face_locations.shape[0]):
                if lables[i]:
                    if lables[i] not in labels_dict:
                        labels_dict[lables[i]] = []
                    labels_dict[lables[i]].append(trackId[i])
            labels_dict = {k: sorted(v) for k, v in labels_dict.items()}
            for k, v in labels_dict.items():
                for duplicate_id in v[:-1]:
                    if duplicate_id in recorder.Identified_ids:
                        recorder.Identified_ids.remove(duplicate_id)
                        indices = np.where(trackId == duplicate_id)[0][0]
                        lables[indices]=None

    """___________________duplicate value check____________________"""

    """___________________End labelling______________________"""

    return lables

