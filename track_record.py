from datetime import datetime
import pandas as pd
import threading
import time
import numpy as np

class TrackRecoder:
    def __init__(self):
        self.Identified_ids = []
        self.Identified_ids_link_person_ids = {}
        self.NonIdentified_track_records = {}
        self.Identified_track_records = []
        self.create_record_file()

    def get_time(self):
        t = datetime.now()
        return f"{t.year:04}-{t.month:02}-{t.day:02}-{t.hour:02}-{t.minute:02}"

    def create_record_file(self):
        t = self.get_time()
        self.file_path  = f"person_track_{t}.csv"
        with open(self.file_path, "w") as file:
            file.write("person_id,x,y,time")

    def update_track(self, records):
        """
        records: list of records
        every record
        record[0]: track id
        record[1]: Identified person id
        recode[2]: point x
        recode[3]: point y
        """
        t = self.get_time()
        
        for record in records:
            id, person_id, x, y = record

            if id not in self.Identified_ids and person_id is not None:
                self.Identified_ids.append(id)
                self.Identified_ids_link_person_ids[id] = person_id
                if id in self.NonIdentified_track_records:
                    id_recodes = np.array(self.NonIdentified_track_records[id])
                    self.NonIdentified_track_records.pop(id)
                    id_recodes[id_recodes==None] = person_id
                    self.Identified_track_records += id_recodes.tolist()
                    del(id_recodes)



            if id in self.Identified_ids:
                person_id = self.Identified_ids_link_person_ids[id]
                self.Identified_track_records.append([person_id, x, y, t])
            elif person_id is None:
                if id not in self.NonIdentified_track_records:
                    self.NonIdentified_track_records[id] = []
                self.NonIdentified_track_records[id].append([person_id, x, y, t])


            


    def autoUpdate(self):
        columns = ["person_id", 'x', 'y', "time"]
        main_df = pd.read_csv(self.file_path)
        temp_df = pd.DataFrame(self.Identified_track_records, columns=columns)
        self.Identified_track_records = []
        main_df = pd.concat([main_df, temp_df], ignore_index=True)
        main_df.to_csv(self.file_path, index=False)
        del(main_df)
        del(temp_df)

    def check_NonIdentified_track_records(self):

        t = datetime.now()
        now_t = t.hour*60 + t.minute
        record_keys = list(self.NonIdentified_track_records.keys())  # create a list of keys
        keys_to_remove = []
        for record in record_keys:
            t = self.NonIdentified_track_records[record][-1][-1].split('-')[-2:]
            t = int(t[0])*60 + int(t[1])
            if abs(now_t - t) > 5:
                keys_to_remove.append(record)  # add key to list of keys to remove

        # remove the keys outside the loop
        for key in keys_to_remove:
            self.NonIdentified_track_records.pop(key)

    def start_timer(self, interval):
        self.interval = interval
        self.timer = threading.Timer(self.interval, self.start_timer, args=[self.interval])
        self.timer.start()
        self.autoUpdate()
        self.check_NonIdentified_track_records()

    def stop_timer(self):
        self.timer.cancel()
