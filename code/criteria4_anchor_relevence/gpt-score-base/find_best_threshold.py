import openai
import os
import openai
import pandas as pd
import os
from openai import OpenAI
import re
import numpy as np
from sklearn.metrics import f1_score


ANCHOR_FILE_PATH = "../../data/example_anchor_answer_info.csv"
VAL_ESSAY_PATH = "../../data/example_article.txt"
GOLD_LABEL_PATH = "gold_labels.txt"
PRED_LABEL_PATH = "tune_info.csv"
BEST_THRESHOLD = "best_threshold_info.txt"
def extract_number_from_string(input_string):
  number = re.findall(r'\d+', input_string)
  if number:
      return int(number[0])
  else:
      print("this is not a number")
  return "empty"



def get_few_shots_chatgpt_response(prompt):
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-4-0613", #"gpt-3.5-turbo" for chatgpt, "gpt-4-0613" for gpt4
    messages = [
        {"role": "system", "content": '''
        Based on the question and the anchor, give a score between 1 to 100 for how confident you are about the question is grounded in anchor sentence. If the question refers to the same entity as the anchor, we consider the question to be grounded.

        '''},
        {"role":"user", "content": prompt}],
    temperature=0, #0
    max_tokens=128
  )

    result = response.choices[0].message.content
    return result

## helper methods
def retrieve_article_by_id(data_df, article_id):
  for i in range(len(data_df)):
    if(data_df.iloc[i][0][0]['ArticleID'] == article_id):
      return data_df.iloc[i][0][0]
    
## this is for random sampling, we sample once and read from it everytime as we do not 
## have that many annotators to annotate all qud for all sentences from one article and do this for all.
## in this example article we just take 1-10 as samples but generally we're doing random sample
def read_sampling_index(df):
    sample_file_path = "../../data/example_sampling.csv"
    sampled_df = pd.read_csv(sample_file_path)
    sampled_indices = list(sampled_df['selected_row_number'] - 1)
    sampled_df = df.loc[sampled_indices].sort_index()

    return sampled_df

def read_context(ESSAY_PATH):
    with open(ESSAY_PATH, 'r') as f:
        essay_lines = f.readlines()
    essay_context = [line.strip().split("\t")[1] for i, line in enumerate(essay_lines)]
    return essay_context


def generate_predict_scores(essay_path):
    df = pd.read_csv(ANCHOR_FILE_PATH)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "index"})

    df = read_sampling_index(df)
    questions = []
    context = read_context(essay_path)
    response = []
    for _, row in df.iterrows():
        question_text = str(row['questions']).replace('<|endoftext|>', '')
        anchor_id = int(row["anchor_id"])
        answer_id = int(row["answer_id"])
        anchor_sentence = context[anchor_id-1]
        prompt = "Question: \n" + question_text +"\n"+ "Anchor Sentence: \n" + anchor_sentence + "\n"
        response.append(get_few_shots_chatgpt_response(prompt))
    response_df = pd.DataFrame({
        "response": response,
        "anchor_id": df["anchor_id"],
        "answer_id": df["answer_id"],
        "questions": df["questions"].replace('', '')
    })    
    response_df['filtered response'] = response_df['response'].apply(lambda val: extract_number_from_string(val))
    
    response_df.to_csv(PRED_LABEL_PATH, index=False)
    print("result saved to " + PRED_LABEL_PATH)

def find_best_thresholds(gold_list, scores_pred):
    gold_list = np.array(gold_list)
    scores_pred = np.array(scores_pred)
    threshold1 = list(range(0, 90, 10))  # Starts from 1 to 90, stops at 91
    threshold2 = list(range(10, 100, 10))  # Starts from 11 to 100, stops at 101

    best_thresh1, best_thresh2, best_f1 = 0, 0, 0

    # Test all combinations of thresholds
    for t1 in threshold1:
        for t2 in threshold2:
            if t1 < t2:  # This ensures t1 is less than t2
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
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.organization = os.getenv('OPENAI_ORGANIZATION')
    generate_predict_scores(VAL_ESSAY_PATH)
    gold_list = pd.read_csv(GOLD_LABEL_PATH, header=None, names=['Number'])['Number'].tolist()#replace the gold list with your 
    pred_df = pd.read_csv(PRED_LABEL_PATH)
    pred_list = pred_df["filtered response"].to_list()
    find_best_thresholds(gold_list=gold_list, scores_pred=pred_list)

if __name__ == "__main__":
    main()

