import openai
import random
import pandas as pd
from nltk import tokenize
import nltk
import os
from openai import OpenAI

ANCHOR_FILE_PATH = "../../data/example_anchor_answer_info.csv"
NP_FILE_OUTPUT_PATH = "example_article_with_np.csv"

def get_chatgpt_response(prompt):
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": "Could you find the longest NP in the sentence provided and the head of longest NP(which should be substring of longest NP) in the following format and return the answer as following format. \
        Longest NP: [] ;\
        Head of Longest NP: []"},
        {"role":"user", "content": prompt}],
    temperature=0, #0
    max_tokens=128
    )

    result = response.choices[0].message.content
    return result


def write_chatgpt_response_back(NP_FILE_OUTPUT_PATH):
    data_df = pd.read_csv(ANCHOR_FILE_PATH)
    question_arr = data_df["questions"].to_list()
    np_arr = []
    np_head_arr = []
    result_arr =[]
    for question in question_arr:
        question = question.split("<|endoftext|>")[0]
        input = "Sentence: [" + question + "]"
        result = get_chatgpt_response(input)
        result_arr.append(result)
        longest_np = result.split(";")[0].split("Longest NP:")[1].strip()[1:-1]
        longest_np_head = result.split(";")[1].split("Head of Longest NP:")[1].strip()[1:-1]
        np_arr.append(longest_np)
        np_head_arr.append(longest_np_head)
    new_df = pd.DataFrame().assign(questions=data_df["questions"], anchor_id=data_df["anchor_id"])
    new_df["Longest np"] = np_arr
    new_df["Longest np head"] = np_head_arr
    new_df["results"] = result_arr
    new_df.to_csv(NP_FILE_OUTPUT_PATH)

def main():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.organization = os.getenv('OPENAI_ORGANIZATION')
    write_chatgpt_response_back(NP_FILE_OUTPUT_PATH)


if __name__ == "__main__":
    main()
