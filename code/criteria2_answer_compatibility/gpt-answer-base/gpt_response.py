import openai
import pandas as pd
import os
from openai import OpenAI
import re
from difflib import SequenceMatcher


ANCHOR_FILE_PATH = "../../data/example_anchor_answer_info.csv"
ESSAY_PATH = "../../data/example_article.txt"

# filter to clean response
def filter_response(input_arr, context):
    result = []
    for answer in input_arr:
        flag = False
        for index, sent in enumerate(context):
            if SequenceMatcher(None, answer, sent).ratio() > 0.65:
              flag = True
              result.append (1)
              break
        
        if not flag:
            result.append (3)

    return result


## gpt response to find the answer and the closest sentence to the answer
def get_chatgpt_response(prompt):
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {"role":"user", "content": prompt}
    ],
    temperature=0, #0
    max_tokens=256
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

def write_response_back():
    df = pd.read_csv(ANCHOR_FILE_PATH)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "index"})

    df = read_sampling_index(df)
    questions = []
    context = read_context(ESSAY_PATH)
    response = []
    for index, row in df.iterrows():
        question_id = index
        question_text = str(row['questions']).replace('<|endoftext|>', '')
        anchor_id = int(row["anchor_id"])
        context_before_include_anchor = context[:(anchor_id)]
        anchor_sentence = context[anchor_id-1]
        prompt_find_answer = "article: " + ''.join(context_before_include_anchor) + "\n" + "After reading sentence:'" + anchor_sentence + "', a reader asked the following question.\n" + "Q:" + question_text + "A:"
        generated_answer = get_chatgpt_response(prompt_find_answer)
        prompt_find_closest_sent = "article:" + ''.join(context) + "\n" + "Which sentence in the article is closest to the sentence: '"  + generated_answer + "'\n" "A:"
        response = get_chatgpt_response(prompt_find_closest_sent)

    response_df = pd.DataFrame({
        "response": response,
        "anchor_id": df["anchor_id"],
        "answer_id": df["answer_id"],
        "questions": df["questions"].replace('', '')
    })
    response_df['filtered response'] = filter_response(response_df['response'], context)
    response_df.to_csv("example_eval.csv")


def main():
    os.environ['OPENAI_API_KEY'] = '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'
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.organization = os.getenv('OPENAI_ORGANIZATION')
    write_response_back()

if __name__ == "__main__":
    main()

