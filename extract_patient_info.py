from data.patient_info import PatientInfo

mirror_id = 2

data_dir = f"./mirror{mirror_id}_data"
#data_dir = "./lab_mirror_data"
output_file = f"overall_patient_info_{mirror_id}.csv"
#output_file = "lab_overall_patient_info.csv"
patient_info = PatientInfo(data_dir, save_dir=output_file, mode="dir")
patient_info_list = patient_info.extract()
patient_info.save()
