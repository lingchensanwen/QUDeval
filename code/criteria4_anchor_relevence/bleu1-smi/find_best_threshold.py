import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import pandas as pd


ANCHOR_FILE_PATH = "../../data/example_anchor_answer_info.csv"
# ANCHOR_FILE_PATH = "/home/yw23374/QUDeval/code/data/example_anchor_answer_info.csv"
ESSAY_PATH = "../../data/example_article.txt"
BEST_THRESHOLD = "best_threshold_info.txt"
GOLD_LABEL_PATH = "gold_labels.txt"
PRED_SCORES_PATH = "computed_scores.txt"


def read_context(essay_path):
    with open(essay_path, 'r') as f:
        essay_lines = f.readlines()
    essay_context = [line.strip().split("\t")[1] for i, line in enumerate(essay_lines)]
    return essay_context
  

def compute_bleu_scores(df):
    print(df.head())
    anchor_ids = df['anchor_id'].to_list()
    questions = df['questions'].to_list()

    contexts = read_context(ESSAY_PATH)
    anchor_sentences = [contexts[anchor_id-1] for anchor_id in anchor_ids]

    BLEU1 = []
    for i in tqdm(range(len(anchor_sentences)), desc="Computing BLEU-1 scores"):
        reference = anchor_sentences[i].split() # the reference sentences are tokenized and placed in a list
        if len(str(questions[i])) == 0:
            candidate = [""]
        else:
            candidate = questions[i].strip().split("<|endoftext|>")
            candidate = candidate[0].split() # the candidate sentences are tokenized
        
        score = sentence_bleu([reference], candidate, weights=(1.0, 0, 0, 0)) # we specify weights to calculate only BLEU-1
        BLEU1.append(score)
    
    with open(PRED_SCORES_PATH, 'w') as file:
        for score in BLEU1:
            file.write(str(score) + '\n')

    return BLEU1


def find_best_sim_thresholds(gold_list, scores_pred):
    gold_list = np.array(gold_list)
    scores_pred = np.array(scores_pred)

    # thresholds
    threshold1 = np.linspace(0, 1, 101)  # Starts from 0 to 1, increment by 0.01
    threshold2 = np.linspace(0, 1, 101)  # Starts from 0 to 1, increment by 0.01

    best_thresh1, best_thresh2, best_f1 = 0, 0, 0

    # Test all combinations of thresholds
    for t1 in threshold1:
        for t2 in threshold2:
            if t1 < t2:  
                # Make predictions based on thresholds
                # Note for criteria 4, 1 is well grounded, 2 is part grounded and 3 is not grounded at all
                predictions = np.where(scores_pred > t2, 1, (np.where((scores_pred > t1) & (scores_pred <= t2), 2, 3)))

                # Calculate macro F1 score
                f1 = f1_score(gold_list, predictions, average='macro')
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh1 = t1
                    best_thresh2 = t2

    text_output = f'Best thresholds: {best_thresh1}, {best_thresh2} with Macro F1 score of {best_f1}'
    with open(BEST_THRESHOLD, 'w') as file:
        file.write(text_output)

def main():
    df = pd.read_csv(ANCHOR_FILE_PATH)
    compute_bleu_scores(df)
    gold_label_list = pd.read_csv(GOLD_LABEL_PATH, header=None, names=['Number'])['Number'].tolist()
    pred_score_list = pd.read_csv(PRED_SCORES_PATH, header=None, names=['Number'])['Number'].tolist()
    find_best_sim_thresholds(gold_label_list, pred_score_list)


if __name__ == "__main__":
    main()


