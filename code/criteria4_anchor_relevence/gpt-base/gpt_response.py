import openai
import pandas as pd
from nltk import tokenize
import os
from openai import OpenAI
import re


ANCHOR_FILE_PATH = "../../data/example_anchor_answer_info.csv"
ESSAY_PATH = "../../data/example_article.txt"

### clean method

def filter_response(input_arr):
    options = ['The question is fully grounded in the anchor sentence.', 'Some parts of the question are grounded in the anchor sentence.', 'The question is not grounded at all in the anchor sentence.']
    def extract_number_from_string(input_string):
        number = re.findall(r'\d+', input_string)
        if number and int(number[0]) in [1, 2, 3]:
            return int(number[0])
        else:
            return None
    def process_column(val):
        val = str(val)
        best_match = ''
        best_len = 0
        for option in options:
            if re.search(re.escape(option), val, re.IGNORECASE):
                if len(option) > best_len:
                    best_match = option
                    best_len = len(option)
        if best_match == '':
            return None
        return options.index(best_match) + 1
    result = input_arr.apply(lambda val: extract_number_from_string(val) if extract_number_from_string(val) is not None else process_column(val))
    return result

## zero-shot response
def get_chatgpt_response(prompt):
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-4-0613", #"gpt-3.5-turbo" for chatgpt, "gpt-4-0613" for gpt4
    messages = [
        {"role":"user", "content": prompt},
        {"role": "system", "content": '''Is the question well-grounded in the anchor sentence? Please evaluate using the following scale:

1: The question is fully grounded in the anchor sentence.
2: Some parts of the question are grounded in the anchor sentence.
3: The question is not grounded at all in the anchor sentence.

Based on the question and the anchor, please choose one of the above options. If the question refers to the same entity as the anchor, we consider the question to be grounded.
'''},
    ],
    temperature=0, #0
    max_tokens=128
  )

    result = response.choices[0].message.content
    return result

## few-shot response
def get_few_shots_response(prompt):
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-4-0613", #"gpt-3.5-turbo" for chatgpt, "gpt-4-0613" for gpt4
    messages = [
        {"role": "system", "content": '''
Here are a few examples for all cases:

Example 1:
Question: What do lawmakers think about this issue?
Anchor Sentence: U.S. exports of nuclear material cannot be adequately traced from country to country, according to a congressional report.
Result: [1: The question is fully grounded in the anchor sentence.]

Example 2:
Question: What does the report say is the reason for the export ban?
Anchor Sentence:  U.S. exports of nuclear material cannot be adequately traced from country to country, according to a congressional report.
Result: [2: Some parts of the question are grounded in the anchor sentence.]

Example 3:
Question: How much plutonium does a nuclear weapon need?
Anchor Sentence: The report says hundreds of tons of plutonium and highly enriched uranium have accumulated worldwide, mostly from nuclear power generation.
Result: [3: The question is not grounded at all in the anchor sentence.]

Example 4:
Question: What is an example of a situation where a moonlit beach is particularly romantic?
Anchor Sentence: MIAMI - In matters of love, nothing says romance like a moonlit beach.
Result: [1: The question is fully grounded in the anchor sentence.]

Example 5:
Question: What was the world's perception of the horseshoe crab before it became recognized for its importance in the ecosystem?
Anchor Sentence: For most of its history, the world regarded it as junk from the sea.
Result: [2: Some parts of the question are grounded in the anchor sentence.]

Example 6:
Question: What was the historical impact of horseshoe crabs on the fishing industry?
Anchor Sentence: But in the 1990s, their numbers began falling.
Result: [3: The question is not grounded at all in the anchor sentence.]

Is the question well-grounded in the anchor sentence? Please evaluate using the following scale:

1: The question is fully grounded in the anchor sentence.
2: Some parts of the question are grounded in the anchor sentence.
3: The question is not grounded at all in the anchor sentence.

Based on the question and the anchor, please choose one of the above options. If the question refers to the same entity as the anchor, we consider the question to be grounded.

'''},
    {"role":"user", "content": prompt},
    ],
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

def write_response_back(few_shots=False):
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
        answer_id = int(row["answer_id"])
        context_before_include_anchor = context[:(anchor_id)]
        anchor_sentence = context[anchor_id-1]
        answer_sentence = context[answer_id-1]
        prompt = "Question: \n" + question_text +"\n"+ "Anchor: \n" + anchor_sentence
        if few_shots == False:
            response.append(get_chatgpt_response(prompt))
        else:
            response.append(get_few_shots_response(prompt))

    response_df = pd.DataFrame({
        "response": response,
        "anchor_id": df["anchor_id"],
        "answer_id": df["answer_id"],
        "questions": df["questions"].replace('', '')
    })
    response_df['filtered response'] = filter_response(response_df['response'])
    if few_shots == False:
        response_df.to_csv("example_zero_shot_eval.csv")
    else:
        response_df.to_csv("example_few_shots_eval.csv")
    return ""



def main():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.organization = os.getenv('OPENAI_ORGANIZATION')
    write_response_back(few_shots=False)#zero shot
    write_response_back(few_shots=True)#few shot

if __name__ == "__main__":
    main()

