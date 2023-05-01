def get_records(detections, new_tracker_list, person_ids):
    records_dict = {}
    for detect in detections:
        x1, y1, x2, y2 = detect[0]
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2
        records_dict[detect[-1]] = [detect[-1], None, x, y]
    for detect, person_id in zip(new_tracker_list, person_ids):
        if person_id:
            records_dict[detect[-1]][1] = person_id
    return list(records_dict.values())


def update_records(detections, new_tracker_list, person_ids, recorder):
    records = get_records(detections, new_tracker_list, person_ids)
    recorder.update_track(records)



def get_labels(detections, recorder):
    labels = []
    for detect in detections:
        if detect[-1] in recorder.Identified_ids:
            labels.append(recorder.Identified_ids_link_person_ids[detect[-1]])
        else:
            labels.append(str(detect[-1]))
    return labels



# def get_labels(detections, new_tracker_list, person_ids):
#     labels_dict = {}
#     for detect in detections:
#         labels_dict[detect[-1]] = str(detect[-1])
#     for detect, person_id in zip(new_tracker_list, person_ids):
#         if person_id:
#             labels_dict[detect[-1]] = person_id
#     return list(labels_dict.values())
    