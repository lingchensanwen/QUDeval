import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import json


BEST_THRESHOLD = "best_threshold_info.json"
THRESHOLD_TUNING_SET = "../data/example_reference_tuning.csv"

def find_best_thresholds(gold_list, scores_pred, metric = None ):
    gold_list = np.array(gold_list)
    scores_pred = np.array(scores_pred)
    
    # thresholds
    if metric == "gpt":
      threshold1 = list(np.linspace(0,100,10))
      threshold2 = list(np.linspace(0,100,10))

    else:
      threshold1 = list(np.linspace(0,1,100))  # Starts from 0 to 1
      threshold2 = list(np.linspace(0,1,100))  # Starts from 0 to 1

    best_thresh1, best_thresh2, best_f1 = 0, 0, 0

    # Test all combinations of thresholds
    for t1 in threshold1:
        for t2 in threshold2:
            if t1 < t2:  # This ensures t1 is less than t2
                # Make predictions based on thresholds
                predictions = np.where(scores_pred > t2, 1, (np.where((scores_pred > t1) & (scores_pred <= t2), 2, 3)))

                # Calculate macro F1 score
                f1 = f1_score(gold_list, predictions, average='macro')
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh1 = t1
                    best_thresh2 = t2

    return best_thresh1, best_thresh2


# you need to compute this for your respective threshold tuning set
def get_thresholds ():
    df = pd.read_csv(THRESHOLD_TUNING_SET)
    thresholds = {'criteria2': {}, 'criteria3': {}, 'criteria4':{}}
    criteria = ['criteria2', 'criteria3', 'criteria4']
    metrics = ['bleu','bert','meteor','rouge','gpt']

    for c in criteria:
        for m in metrics:
            t1, t2 = find_best_thresholds (df[c].astype('int'), df[m], m)
            dict = {"threshold1":t1, "threshold2":t2}
            thresholds[c][m] = dict

    with open(BEST_THRESHOLD, 'w') as file:
        json.dump(thresholds, file)

def main():
    get_thresholds()

if __name__ == "__main__":
    main()