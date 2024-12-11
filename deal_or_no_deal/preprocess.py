import re
import json


def extract_numbers(input_file_path, output_json_file_path):
    with open(input_file_path, "r") as file:
        lines = file.readlines()

    skip_next_line = False
    for idx, line in enumerate(lines):
        # to prevent repeats
        if skip_next_line:
            skip_next_line = False
            continue
        if "<disagree>" not in line:
            skip_next_line = True
        print(idx)
        input_numbers = []
        partner_input_numbers = []
        line = line.strip()
        input_match = re.search(
            r"<input>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</input>", line
        )
        partner_input_match = re.search(
            r"<partner_input>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</partner_input>",
            line,
        )

        if input_match and partner_input_match:
            for i in range(1, 7):
                input_numbers.append(int(input_match.group(i)))
                partner_input_numbers.append(int(partner_input_match.group(i)))

            quantities_A = input_numbers[0::2]
            values_A = input_numbers[1::2]
            quantities_B = partner_input_numbers[0::2]
            values_B = partner_input_numbers[1::2]
            assert quantities_A == quantities_B
            quantities = quantities_A
            datapoint = {
                "quantities": quantities,
                "values_A": values_A,
                "values_B": values_B,
            }

            with open(output_json_file_path, "a") as output_file:
                json.dump(datapoint, output_file)
                output_file.write("\n")

    return


path_prefix = r"C:\Users\Pranava\Documents\Stanford\course_materials\cs329x_hci_nlp\project\moderation_protocol\deal_or_no_deal\\"
input_train_txt = path_prefix + "train.txt"
output_train_json = path_prefix + "train.json"
extract_numbers(
    input_file_path=input_train_txt, output_json_file_path=output_train_json
)

input_train_txt = path_prefix + "train.txt"
output_train_json = path_prefix + "train.json"
extract_numbers(
    input_file_path=input_train_txt, output_json_file_path=output_train_json
)

input_val_txt = path_prefix + "val.txt"
output_val_json = path_prefix + "val.json"
extract_numbers(input_file_path=input_val_txt, output_json_file_path=output_val_json)

input_test_txt = path_prefix + "test.txt"
output_test_json = path_prefix + "test.json"
extract_numbers(input_file_path=input_test_txt, output_json_file_path=output_test_json)
