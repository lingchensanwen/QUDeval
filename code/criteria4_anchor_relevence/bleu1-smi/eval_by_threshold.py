
import pandas as pd
import re
from sklearn.metrics import f1_score


ANCHOR_FILE_PATH = "../../data/example_anchor_answer_info.csv"
VAL_ESSAY_PATH = "../../data/example_article.txt"
PRED_SCORES_PATH = "computed_scores.txt"
BEST_THRESHOLD = "best_threshold_info.txt"
PRED_LABEL_PATH = "pred_labels.txt"

def convert_score_to_label_on_threshold(score, best_thresh1, best_thresh2):
    if score > best_thresh2:
        return 1 #1 is well grounded
    elif score > best_thresh1:
        return 2 #2 is part grounded
    else:
        return 3 #3 is not grounded

def main():
    ## let's re-use predict scores generated before, in your case, please use different set to find threshold and make prediction on your test set
    pred_score_list = pd.read_csv(PRED_SCORES_PATH, header=None, names=['Number'])['Number'].tolist()
    with open(BEST_THRESHOLD, 'r') as file:
        line = file.readline().strip()

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    best_thresh1, best_thresh2 = map(float, numbers[:2])
    print(f"best threshold 1 and 2 read is {best_thresh1} and {best_thresh2}")
    pred_labels = [convert_score_to_label_on_threshold(score, best_thresh1, best_thresh2) for score in pred_score_list]
    with open(PRED_LABEL_PATH, 'w') as file:
        for score in pred_labels:
            file.write(str(score) + '\n')


if __name__ == "__main__":
    main()

