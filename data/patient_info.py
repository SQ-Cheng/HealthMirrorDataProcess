import os
import json
import csv

class PatientInfo:
    def __init__(self, data_dir, save_dir="patient_info.csv", mode="file"):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.patient_info_list = None
        self.mode = mode  # "file" or "dir"

    def extract(self, data_dir=None, data_file=None):
        if self.mode == "file":
            return self._file_extract(data_file)
        elif self.mode == "dir":
            return self._dir_extract(data_dir)

    def _file_extract(self, data_file):
        self.patient_info_list = []
        with open(data_file, 'r') as f:
            reader = csv.DictReader(f, fieldnames=['lab_patient_id', 'hospital_patient_id', 'blood_oxygen', 'heart_rate', 'respiratory_rate', 'temperature', 'low_blood_pressure', 'high_blood_pressure'])
            for row in reader:
                self.patient_info_list.append(row)
        return self.patient_info_list

    def _dir_extract(self, data_dir=None):
        if data_dir is None:
            data_dir = self.data_dir
        patient_info_list = []
        for dir in os.listdir(data_dir):
            dir_path = os.path.join(data_dir, dir)
            if os.path.isdir(dir_path):
                patient_file = os.path.join(dir_path, "patient_info.txt")
                try:
                    with open(patient_file, 'r') as f:
                        for line in f:
                            if line.startswith('Patient ID: '):
                                lab_patient_id = line[len('Patient ID:'):].strip()
                            elif line.startswith('Patient Info:'):
                                json_str = line[len('Patient Info:'):].strip().strip('"').replace('\\"', '"').replace('\\/', '/')
                                try:
                                    patient_info = json.loads(json_str)
                                    hospital_patient_id = patient_info.get('patient_id', 'n/a')
                                    vital_str = patient_info.get('vitals', {})
                                    vital_dict = {
                                        'blood_oxygen': vital_str.get('blood_oxygen', 'n/a'),
                                        'heart_rate': vital_str.get('heart_rate', 'n/a'),
                                        'respiratory_rate': vital_str.get('respiratory_rate', 'n/a'),
                                        'temperature': vital_str.get('temperature', 'n/a'),
                                        'blood_pressure': vital_str.get('blood_pressure', 'n/a'),
                                    }
                                    spo2 = int(vital_dict['blood_oxygen'].strip('%') if vital_dict['blood_oxygen'] != 'n/a' else -1)
                                    hr = int(vital_dict['heart_rate'].strip('bpm') if vital_dict['heart_rate'] != 'n/a' else -1)
                                    rr = int(vital_dict['respiratory_rate'].strip('bpm') if vital_dict['respiratory_rate'] != 'n/a' else -1)
                                    temp = float(vital_dict['temperature'][:-2] if vital_dict['temperature'] != 'n/a' else -1)
                                    bp = (vital_dict['blood_pressure']).split('/') if vital_dict['blood_pressure'] != 'n/a' else 'n/a'
                                    hbp = int(bp[0]) if bp != 'n/a' and len(bp) > 1 else -1
                                    lbp = int(bp[1]) if bp != 'n/a' and len(bp) > 1 else -1
                                    patient_info_list.append({
                                        'lab_patient_id': lab_patient_id,
                                        'hospital_patient_id': hospital_patient_id,
                                        'blood_oxygen': spo2,
                                        'heart_rate': hr,
                                        'respiratory_rate': rr,
                                        'temperature': temp,
                                        'low_blood_pressure': lbp,
                                        'high_blood_pressure': hbp
                                    })
                                except Exception as e:
                                    print(f"Error decoding JSON from {patient_file}: {e}")
                                    continue
                except Exception as e:
                    print(f"Error reading {patient_file}: {e}")
                    continue

        # show how many individual hospital_patient_ids are extracted
        unique_hospital_ids = set(info['hospital_patient_id'] for info in patient_info_list)
        print(f"Extracted information for {len(unique_hospital_ids)} unique hospital patient IDs.")
        self.patient_info_list = patient_info_list
        return patient_info_list
    
    def save(self, output_file=None):
        if output_file is None:
            output_file = self.save_dir
        fieldnames = ['lab_patient_id', 'hospital_patient_id', 'blood_oxygen', 'heart_rate', 'respiratory_rate', 'temperature', 'low_blood_pressure', 'high_blood_pressure']
        if self.patient_info_list is None:
            print("Run extract() first.")
            return
        try:
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for patient_info in self.patient_info_list:
                    writer.writerow(patient_info)
            print(f"Patient information saved to {output_file}")
        except Exception as e:
            print(f"Error writing to CSV file {output_file}: {e}")