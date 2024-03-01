import openai
import pandas as pd
import os
from openai import OpenAI
import re


ANCHOR_FILE_PATH = "../../data/example_anchor_answer_info.csv"
ESSAY_PATH = "../../data/example_article.txt"

## filter to clean response
def filter_response(input_arr):
    result = []
    options = ['No new concept', 'Answer leakage', 'Hallucination']

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
    model="gpt-3.5-turbo",
    messages = [
        {"role":"user", "content": prompt},
        {"role": "system", "content": '''Does the question contain new concepts that a reader would be hard to come up with? (By "new concepts", I mean concepts that cannot be easily inferred by world knowledge from existing ones). There are several possibilities here as well:
    This question does not contain new concepts.
    Answer leakage: The question contains new concepts that are in the answer sentence AND not in the context.
    Hallucination: The question contains new concepts. This includes:
    Concepts not in the article.
    The question contains new concepts that are not in the context, but can be found later in the document.

    Given the Context, Question, and Answer, select one of the following options on the basis of your understanding of the instructions.
    1: No new concepts
    2: Answer leakage
    3: Hallucination
    '''},
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
        prompt = "Context: \n" + '\n'.join(context_before_include_anchor) + "\n\n" + "Question: \n" + question_text +"\n"+ "Answer: \n" + answer_sentence + "\n"
        if few_shots == False:
            response.append(get_chatgpt_response(prompt))
        else:
            response.append(get_few_shots_chatgpt_response(prompt))

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

## few-shot response
def get_few_shots_chatgpt_response(prompt):
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": '''
    Here are a few examples for all cases:
    Example 1:
    Context:
    1 U.S. exports of nuclear material cannot be adequately traced from country to country, according to a congressional report.
    Question:
    What does the report say is the reason for the export ban?
    Answer Sentence:
    The report says hundreds of tons of plutonium and highly enriched uranium have accumulated worldwide, mostly from nuclear power generation.
    Selected option:
    [3: Hallucination]

    Example 2:
    Context:
    1 U.S. exports of nuclear material cannot be adequately traced from country to country, according to a congressional report.
    2 'Scarcely a day goes by without a report of a new black market deal,' said Sen. John Glenn in a statement reacting to the report.
    3 'Given the staggering amount of nuclear materials we have exported, it could only be a matter of time before some of this deadly contraband proves to be of U.S.
    4 origin.'
    5 As chairman of the Senate Committee on Governmental Affairs in the last Congress, Glenn commissioned the report from the General Accounting Office, which conducts investigations for legislators.
    6 The report says hundreds of tons of plutonium and highly enriched uranium have accumulated worldwide, mostly from nuclear power generation.
    7 It does not include figures on U.S. nuclear exports but says 71 export licenses for nuclear materials were granted in 1993.
    8 Nuclear exports for weapons use or weapons research are prohibited, as is transfer of nuclear materials to a third country.
    9 The report said U.S. tracking showed that Japan produced about 20.3 metric tons of plutonium from U.S. materials between 1978 and 1992, but Japanese records showed 58.7 tons.
    Question:
    How much plutonium does a nuclear weapon need?
    Answer Sentence:
    A nuclear weapon can be made with as little as 6 kilograms (13.2 pounds) of plutonium, U.S. officials have said.
    Selected option:
    [2: Answer leakage]

    Example 3:
    Context:
    1 U.S. exports of nuclear material cannot be adequately traced from country to country, according to a congressional report.
    Question:
    What do lawmakers think about this issue?
    Answer Sentence:
    'Scarcely a day goes by without a report of a new black market deal,' said Sen. John Glenn in a statement reacting to the report.
    Selected option:
    [1: No new concepts]

    Example 4:
    Context:
    1 MIAMI - In matters of love, nothing says romance like a moonlit beach.
    2 Especially if you're a lusty horseshoe crab and the tide is high.
    3 Every spring, from Florida to New Jersey, crabs that look more like fossils than a postcard for passion make their way ashore by the thousands when the moon is bright to lay millions of eggs that provide critical food for migrating shorebirds.
    4 But in the 1990s, their numbers began falling.
    Question:
    What was the historical impact of horseshoe crabs on the fishing industry?
    Answer Sentence:
    And in the mid-1800s, between 1.5 and 2 million were caught yearly to use as fertilizer and livestock feed, according to the Atlantic States Marine Fisheries Commission.
    Selected option:
    [3: Hallucination]

    Example 5:
    Context:
    1 MIAMI - In matters of love, nothing says romance like a moonlit beach.
    2 Especially if you're a lusty horseshoe crab and the tide is high.
    3 Every spring, from Florida to New Jersey, crabs that look more like fossils than a postcard for passion make their way ashore by the thousands when the moon is bright to lay millions of eggs that provide critical food for migrating shorebirds.
    Question:
    What happened to the numbers of horseshoe crabs in the 1990s?
    Answer Sentence:
    But in the 1990s, their numbers began falling.
    Selected option:
    [2: Answer leakage]

    Example 6:
    Context:
    1 MIAMI - In matters of love, nothing says romance like a moonlit beach.
    Question:
    What is an example of a situation where a moonlit beach is particularly romantic?
    Answer Sentence:
    Especially if you're a lusty horseshoe crab and the tide is high.
    Selected option:
    [1: No new concepts]

    Does the question contain new concepts that a reader would be hard to come up with? (By "new concepts", I mean concepts that cannot be easily inferred by world knowledge from existing ones). There are several possibilities here as well:
    This question does not contain new concepts.
    Answer leakage: The question contains new concepts that are in the answer sentence AND not in the context.
    Hallucination: The question contains new concepts. This includes:
    Concepts not in the article.
    The question contains new concepts that are not in the context, but can be found later in the document.

    Given the Context, Question, and Answer, select one of the following options on the basis of your understanding of the instructions.
    1: No new concepts
    2: Answer leakage
    3: Hallucination

    '''},
    {"role":"user", "content": prompt},
    ],
    temperature=0, #0
    max_tokens=128
    )

    result = response.choices[0].message.content
    return result





def main():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.organization = os.getenv('OPENAI_ORGANIZATION')
    write_response_back(few_shots=False)#zero shot
    write_response_back(few_shots=True)#few shot

if __name__ == "__main__":
    main()

