import pandas as pd
import numpy as np

ENTITY_FILE_PATH = "example_entity.txt"
RULE_BASE_FILE_PATH = "/home/yw23374/QUDeval/code/criteria3_giveness/rule-base/example_content-based_strict_False.txt"

def process_entity(entity_file, rule_base_file):

    mapped_produced_list = []
    yufang_response = pd.read_csv(entity_file,sep="\t", header=None)
    rule_based_df = pd.read_csv(rule_base_file, sep="\t", header=None)

    rule_based_refer=  [str(num) for num in rule_based_df[0]]
    print(len(rule_base_file))
    produced_list = yufang_response[1].to_list()

    for idx in range(len(produced_list)):
        element = produced_list[idx]
        #if old(0)/meditated(2) -> no new concept
        if element == 0 or element == 2:
            mapped_produced_list.append('1')
            #if new(1) -> use rule base to tell if that's answer leakage or mediated
        else:
            mapped_produced_list.append(rule_based_refer[idx])
    return mapped_produced_list

def main():
    result = process_entity(ENTITY_FILE_PATH, RULE_BASE_FILE_PATH)
    np.savetxt(f"hou_output.txt", result, delimiter=',', fmt='%s')
    

if __name__ == "__main__":
    main()
