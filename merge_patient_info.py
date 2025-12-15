import csv

def merge_patient_info(extracted_file, marked_file, output_file):
    with open(extracted_file, 'r', encoding='utf-8') as ef, \
         open(marked_file, 'r', encoding='utf-8') as mf, \
         open(output_file, 'w', encoding='utf-8', newline='') as of:
        extracted_reader = csv.reader(ef)
        marked_reader = csv.reader(mf)
        output_writer = csv.writer(of)

        next(extracted_reader)
        for extracted_row in extracted_reader:
            next(marked_reader)
            for marked_row in marked_reader:
                output_row = extracted_row.copy()
                if int(extracted_row[0]) == int(marked_row[0]):
                    for i in range(1, len(extracted_row)):
                        if extracted_row[i] != '-1':
                            output_row[i] = extracted_row[i]
                        elif marked_row[i] != '-1':
                            output_row[i] = marked_row[i]
                    output_writer.writerow(output_row)
                    break

            mf.seek(0)

def main():
    lab = False
    if lab:
        extracted_file = 'lab_overall_patient_info.csv'
        marked_file = 'lab_overall_patient_info.csv'
        output_file = 'lab_merged_patient_info.csv'
    else:
        extracted_file = 'overall_patient_info.csv'
        marked_file = 'extracted_vitals_1.csv'
        output_file = 'merged_patient_info.csv'
    merge_patient_info(extracted_file, marked_file, output_file)

if __name__ == '__main__':
    main()
