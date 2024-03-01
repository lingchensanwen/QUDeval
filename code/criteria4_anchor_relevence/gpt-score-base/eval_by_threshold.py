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
PRED_LABEL_PATH = "tune_info_labels.csv"
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


def generate_predict_labels(essay_path):
    with open(BEST_THRESHOLD, 'r') as file:
        line = file.readline().strip()

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    best_thresh1, best_thresh2 = map(float, numbers[:2])
    print(f"best threshold 1 and 2 read is {best_thresh1} and {best_thresh2}")

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
        result = get_few_shots_chatgpt_response(prompt)
        score = extract_number_from_string(result)
        if score > best_thresh2:
            response.append(1)
        elif score > best_thresh1:
            response.append(2)
        else:
            response.append(3)

    response_df = pd.DataFrame({
        "response": response,
        "anchor_id": df["anchor_id"],
        "answer_id": df["answer_id"],
        "questions": df["questions"].replace('', '')
    })    

    response_df.to_csv(PRED_LABEL_PATH, index=False)
    print("result saved to " + PRED_LABEL_PATH)

def main():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.organization = os.getenv('OPENAI_ORGANIZATION')
    generate_predict_labels(VAL_ESSAY_PATH)

if __name__ == "__main__":
    main()

