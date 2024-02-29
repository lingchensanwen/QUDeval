## import statements
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import spacy
import pandas as pd
import openai
import numpy as np
import os


# those needs to be downloaded
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('wordnet')
# python -m spacy download en_core_web_sm

ANCHOR_FILE_PATH = "../../data/example_anchor_answer_info.csv"
NP_FILE_PATH = "example_article_with_np.csv"
ESSAY_PATH = "../../data/example_article.txt"


## preprocess data
STOP_WORDS = set(stopwords.words('english'))


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

def get_content_words(text):
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()] # Remove punctuation
    words = [word for word in words if word not in STOP_WORDS] # Remove stop words
    tagged = pos_tag(words)

    # Extract content words: nouns (NN, NNS, NNP, NNPS),
    # verbs (VB, VBD, VBG, VBN, VBP, VBZ),
    # adjectives (JJ, JJR, JJS) and adverbs (RB, RBR, RBS)
    content_words = [word for word, pos in tagged if pos in ('NN', 'NNS', 'NNP', 'NNPS',
                                                             'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                                                             'JJ', 'JJR', 'JJS',
                                                             'RB', 'RBR', 'RBS')]
    return content_words


def get_content_noun_words(text):
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()] # Remove punctuation
    words = [word for word in words if word not in STOP_WORDS] # Remove stop words
    tagged = pos_tag(words)

    content_words = [word for word, pos in tagged if pos in ('NN', 'NNS', 'NNP', 'NNPS')]
    return content_words

def get_chatgpt_response(prompt):
  response = openai.ChatCompletion.create(
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

def is_question_words(word):
    question_words = ["how", "what", "do", "did", "is", "are", "was", "were", "will", "can", "could", "should", "would"]
    word = word.lower()
    if word in question_words:
        return True
    return False

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


def contend_based_criteria3_strict(question, answer, context):
  question_content_words = lemmatize_words([word.lower() for word in get_content_noun_words(question)])
  answer_content_words = lemmatize_words([word.lower() for word in get_content_noun_words(answer)])
  context_content_words = lemmatize_words([word.lower() for word in get_content_noun_words(context)])


  # common_words = set(question_content_words).intersection(set(answer_content_words))
  for word in question_content_words:
    if is_question_words(word) == True:
      continue
    if is_pronoun_or_human_name(word) == False:
      if word not in set(context_content_words) and word not in set(context.split()):#not in context or context content words
        return "2"
  return "1"


def contend_based_criteria3(question, answer, context):
  result = "1"
  question_content_words = lemmatize_words([word.lower() for word in get_content_noun_words(question)])
  answer_content_words = lemmatize_words([word.lower() for word in get_content_noun_words(answer)])
  context_content_words = lemmatize_words([word.lower() for word in get_content_noun_words(context)])
  context_lemmatized = lemmatize_words([word.lower() for word in context.split()])
  answer_lemmatized = lemmatize_words([word.lower() for word in answer.split()])

  # common_words = set(question_content_words).intersection(set(answer_content_words))
  for word in question_content_words:
    if is_question_words(word) == True:
      continue
    if is_pronoun_or_human_name(word) == False:
      if word not in set(context_content_words) and word not in set(context_lemmatized):#not in context or context content words
        if word in set(answer_content_words) or word in set(answer_lemmatized):
          result = "2" #answer leakage
        else:
          return "3" #hallunication
  return result

def np_based_criteria3_strict(question, answer, context, longest_np):
  longest_np_lemmatized = lemmatize_words([word.lower() for word in str(longest_np).split()])
  context_lemmatized = lemmatize_words([word.lower() for word in context.split()])
  answer_lemmatized = lemmatize_words([word.lower() for word in answer.split()])
  common_words = set(longest_np_lemmatized).intersection(set(answer_lemmatized))

  for word in common_words:
    if is_question_words(word) == True:
      continue
    if is_pronoun_or_human_name(word) == False:
      if word not in set(context_lemmatized):
          return "2"
  return "1" # no new concept


def np_based_criteria3(question, answer, context, longest_np):
    result = "1"
    longest_np_lemmatized = lemmatize_words([word.lower() for word in str(longest_np).split()])
    context_lemmatized = lemmatize_words([word.lower() for word in context.split()])
    answer_lemmatized = lemmatize_words([word.lower() for word in answer.split()])
    for word in longest_np_lemmatized:
        if is_question_words(word) == True:
            continue
        if is_pronoun_or_human_name(word) == False:
            if word not in set(context_lemmatized):
              if word in set(answer_lemmatized):
                result = "2" #answer leakage
              else:
                return "3" #hallunication
    return result # no new concept



def check_word_cases(question, answer, context, rule, longest_np, strict):
  if rule == "content-based":
    if strict == True:
      return contend_based_criteria3_strict(question, answer, context)
    return contend_based_criteria3(question, answer, context)
  if rule == "NP-based":
    if strict == True:
      return np_based_criteria3_strict(question, answer, context, longest_np)
    return np_based_criteria3(question, answer, context, longest_np)
  
  
def rule_base_criteria3(rule, strict):

    data = pd.read_csv(NP_FILE_PATH)
    np_list = data["Longest np"].to_list()
    question_list = data["questions"].to_list()

    df = pd.read_csv(ANCHOR_FILE_PATH)

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "index"})

    df = read_sampling_index(df)
    df = df.reset_index()
    questions = []
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
        answer_id = int(df_row["answer_id"])
        context_before_include_anchor = " ".join(context[:(anchor_id)])
        anchor_sentence = context[anchor_id-1]
        answer_sentence = context[answer_id-1]
        longest_np = np_list[question_id]
        result = check_word_cases(question_text, answer_sentence, context_before_include_anchor, rule, longest_np, strict)
        response.append(result)
    return response
  

def main():
    parser = argparse.ArgumentParser(description="Run rule-based criteria with command-line arguments.")
    parser.add_argument("--rule", type=str, required=True, help="The rule to apply.")
    parser.add_argument("--strict", type=lambda x: (str(x).lower() == 'true'), required=True, help="Whether to apply rules strictly.")
    
    args = parser.parse_args()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.organization = os.getenv('OPENAI_ORGANIZATION')
    result = rule_base_criteria3(rule=args.rule, strict=args.strict)
    #rule can be content_based / np_based
    #strict can be True/ False
    np.savetxt(f"example_{args.rule}_strict_{args.strict}.txt", result, delimiter=',', fmt='%s')
    

if __name__ == "__main__":
    main()