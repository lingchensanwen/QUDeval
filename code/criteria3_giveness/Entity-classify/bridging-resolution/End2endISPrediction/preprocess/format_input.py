import spacy
import pandas as pd
import nltk

ESSAY_PATH = "/home/yw23374/QUDeval/code/data/example_article.txt"
NP_FILE_PATH = "example_article_with_np.csv"
TO_CLASSIFY_OUTPUT_FILE = "example_article_to_classify.tsv"

def is_pronoun(word):
    if word is None or pd.isnull(word): # check for None or NaN values
      return False
    tag = nltk.pos_tag([word])[0][1]
    return tag in ['PRP', 'PRP$', 'WP', 'WP$']

def is_mention_prev(context, mention, mention_head):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(context)
    noun_phrases = list(doc.noun_chunks)
    np_list = [np.text for np in noun_phrases]
    np_head_list = [np.root.text for np in noun_phrases]

    str_result1 = "FALSE"
    str_result2 = "FALSE"

    if is_pronoun(mention):
        str_result1 = "UNKNOWN"
    elif mention in np_list:
        str_result1 = "TRUE"

    if is_pronoun(mention_head):
        str_result2 = "UNKNOWN"
    elif mention_head in np_head_list:
        str_result2 = "TRUE"

    return str_result1 + ", " + str_result2


def generate_article(num, essay_path):
    article = pd.read_csv(essay_path, sep="\t", header=None)

    sentences = article[1].to_list()
    trimmed_sentences = []
    for sentence in sentences:
        if ' ' in sentence:  # make sure the sentence has at least one space
            trimmed_sentence = sentence[sentence.index(' ') + 1:]
            trimmed_sentences.append(trimmed_sentence)

    return trimmed_sentences[:num]

def get_prev_mention_arr(df, essay_path):
    prev_mention_arr = []
    for idx in range(len(df)):
        anchor_id = df["anchor_id"][idx]
        if pd.isna(anchor_id):
            return ""
        context = " ".join(generate_article(int(anchor_id), essay_path))
        mention = df["Longest np"][idx]
        mention_head = df["Longest np head"][idx]
        prev_mention_arr.append(is_mention_prev(context, mention, mention_head))
    return prev_mention_arr

def add_tokens_to_string(main_string, substring):
    # tokens to be added
    token1 = "[unused1] "
    token2 = " [unused2]"

    if substring in main_string:
        # if substring exists in main string, add tokens before and after the substring
        main_string = main_string.replace(substring, f"{token1}{substring}{token2}")
    else:
        # if substring doesn't exist, add tokens before and after the main string
        main_string = f"{token1}{main_string}{token2}"

    return main_string


def main():
    df = pd.read_csv(NP_FILE_PATH)
    mention_arr = get_prev_mention_arr(df, ESSAY_PATH)
    sent_with_token_arr = []
    span_arr = df["Longest np"].to_list()
    question_arr = [(str(question).replace('<|endoftext|>','').replace('\n','')) for question in df["questions"].to_list()]
    for question, mention in zip(question_arr, span_arr):
        sent_with_token_arr.append(add_tokens_to_string(str(question), mention))

    new_df = pd.DataFrame(
        {"category": len(df)*["new_np"],
        "doc_str": len(df)*["filename"],
        "padding": mention_arr,
        "sent_with_token": sent_with_token_arr,
        "span":  span_arr,
        "placeholder": len(df)*["unknown"]}
    )
    new_df.to_csv(TO_CLASSIFY_OUTPUT_FILE, sep="\t", index=False, header=None)   


if __name__ == "__main__":
    main()
