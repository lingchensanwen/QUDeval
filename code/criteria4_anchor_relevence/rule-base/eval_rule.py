import openai
import os
import openai
import pandas as pd
import os
from openai import OpenAI
import re
import numpy as np
from sklearn.metrics import f1_score
import spacy
import nltk
from nltk.stem import WordNetLemmatizer

NP_FILE_PATH = "example_article_with_np.csv"
ESSAY_PATH = "../../data/example_article.txt"
PRED_LABEL_PATH = "rule_pred.csv"

def is_pronoun_or_human_name(word):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(word)
    if len(doc) == 1:
        token = doc[0]
        if token.pos_ == "PRON" or token.ent_type_ == "PERSON":
            return True

    return False



def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


## helper methods
def is_question_words(word):
    question_words = ["how", "what", "do", "did", "is", "are", "was", "were", "will", "can", "could", "should", "would"]
    word = word.lower()
    if word in question_words:
        return True
    return False


def np_based_criteria4(anchor, longest_np, longest_np_head):
    longest_np_lemmatized = set(lemmatize_words([word.lower() for word in str(longest_np).split()]))
    longest_np_head_lemmatized = set(lemmatize_words([word.lower() for word in str(longest_np_head).split()]))
    anchor_lemmatized = set(lemmatize_words([word.lower() for word in anchor.split()]))

    not_ground_flag = False
    head_in_anchor_flag = all(word in anchor_lemmatized for word in longest_np_head_lemmatized)

    for word in longest_np_lemmatized:
        if is_question_words(word) == True:
            continue
        if is_pronoun_or_human_name(word) == False:
            if word not in anchor_lemmatized:
              not_ground_flag = True # not grounded
            if not_ground_flag and word in anchor_lemmatized:
              return "2" # some grounded
            if not_ground_flag and head_in_anchor_flag:
              return "2" # some grounded

    if not_ground_flag:
      return "3" # not grounded

    return "1" # fully grounded


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


def rule_base_criteria4():
    df = pd.read_csv(NP_FILE_PATH)
    np_list = df["Longest np"].to_list()
    head_np_list = df["Longest np head"].to_list()
    question_list = df["questions"].to_list()

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "index"})

    df = read_sampling_index(df)
    context = read_context(ESSAY_PATH)
    response = []

    for (df_index, df_row), question_from_list in zip(df.iterrows(), question_list):
        question_id = df_index
        question_from_list = str(question_from_list).replace('', '')
        question_text = str(df_row['questions']).replace('', '')
        if question_from_list != question_text:
            print(question_from_list)
            print(question_text)
            print("not match!")
            return
        anchor_id = int(df_row["anchor_id"])
        anchor_sentence = context[anchor_id-1]
        longest_np = np_list[question_id]
        longest_np_head = head_np_list[question_id]
        result = np_based_criteria4(anchor_sentence, longest_np, longest_np_head)
        response.append(result)
    response_df = pd.DataFrame({
        "response": response,
        "anchor_id": df["anchor_id"],
        "questions": df["questions"].replace('', '')
    })    

    response_df.to_csv(PRED_LABEL_PATH, index=False)
    print("result saved to " + PRED_LABEL_PATH)

def main():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.organization = os.getenv('OPENAI_ORGANIZATION')
    rule_base_criteria4()

if __name__ == "__main__":
    main()
