import openai
import pandas as pd
import os
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu
from evaluate import load
from rouge_score import rouge_scorer
import json
from sklearn.metrics import f1_score, classification_report


DATA_PATH = "../data/example_reference_data.csv"
BEST_THRESHOLD = "best_threshold_info.json"
bertscore = load("bertscore")
meteor = load('meteor')
rouge = rouge_scorer.RougeScorer(['rouge1'])


def get_chatgpt_response(prompt):
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {"role":"user", "content": prompt}
    ],
    temperature=0, #0
    max_tokens=128
    )

    result = response.choices[0].message.content
    return result

def compute_bleu (reference, candidate):
    bleuScore = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))
    print (bleuScore)
    return round (bleuScore, 2)

def compute_bertscore (reference, candidate):
    bertScore = bertscore.compute(predictions=[candidate], references=[reference], lang = "en", rescale_with_baseline = True)
    print ("bert", round (bertScore["f1"][0], 2))
    return round (bertScore["f1"][0], 2)

def compute_meteor (reference, candidate):
    meteor_score = meteor.compute(predictions=[candidate], references=[reference])

    return round(meteor_score['meteor'], 2)

def compute_rouge (reference, candidate):
    precision, recall, fmeasure = rouge.score(candidate, reference)['rouge1']

    return round (fmeasure, 2)

def compute_gpt_score (reference, candidate):
    prompt = "Reference Question: " + reference + "\n" + "Candidate Question" + candidate + "\n\n" + "Score the above candidate question for similarity with respect to the reference question on a continuous scale from 0 to 100, where a score of zero means 'no similarity' and a score of one hundred means 'similar intent and phrasing'." + "Score:"
    response = get_chatgpt_response(prompt)

    return response


def compute_all_scores ():
    df = pd.read_csv(DATA_PATH)
    result = {'bleu': [], 'bert': [], 'meteor': [], 'rouge': [], 'gpt': []}

    for index, row in df.iterrows():
        gold_question = str(row['gold question']).replace('<|endoftext|>', '')
        generated_question = row['generated question']
        print (gold_question, generated_question)
        result["bleu"].append (compute_bleu(gold_question, generated_question))
        result["bert"].append (compute_bertscore(gold_question, generated_question))
        result["rouge"].append (compute_rouge(gold_question, generated_question))
        result["meteor"].append (compute_meteor(gold_question, generated_question))
        result["gpt"].append (compute_gpt_score(gold_question, generated_question))
   
    response_df = df
    df['bleu'] = result['bleu']
    df['bert'] = result['bert']
    df['meteor'] = result['meteor']
    df['rouge'] = result['rouge']
    df['gpt'] = result['gpt']
    
    print (response_df)
    response_df.to_csv("example_eval.csv")


def compute_f1():
    df = pd.read_csv('example_eval.csv')
    f = open(BEST_THRESHOLD)
    thresholds = json.load(f)
    m = 'gpt' #specify the reference based evaluation metric
    c = 'criteria4' # specify the evaluation criterion
    threshold1 = thresholds[c][m]['threshold1']
    threshold2 = thresholds[c][m]['threshold2']
    f1 = []

    pred_scores = []
    for id, row in df.iterrows():
        if (row[m] >= threshold2):
            pred_scores.append (1)

        elif ((row[m] < threshold2) & (row[m] > threshold1)):
            pred_scores.append (2)

        else:
            pred_scores.append (3)

    f1_all = f1_score(df[c].astype('int'), pred_scores, average='macro')
    f1_1 = f1_score(df[c].astype('int'), pred_scores, average = 'macro', labels=['1'])
    f1_2 = f1_score(df[c].astype('int'), pred_scores, average = 'macro', labels=['2'])
    f1_3 = f1_score(df[c].astype('int'), pred_scores, average = 'macro', labels=['3'])
    f1.append ([m, round (f1_1, 2), round (f1_2, 2), round (f1_3, 2), round (f1_all, 2)])
    print (f1, classification_report(df[c].astype('int'), pred_scores, target_names=['1', '2', '3']))



def main():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.organization = os.getenv('OPENAI_ORGANIZATION')
    compute_all_scores()
    compute_f1()

if __name__ == "__main__":
    main()
