
In this implementation, there are many approaches to call GPT APIs (also some not calling though), so we save multiple files in the code to make paying for the API meaningful and easy to debug. For each criterion, please go to the corresponding directory and check its readme file. For your implementation, you are expected to set up file paths for desired inputs and outputs. We are still actively modifying and cleaning the code, and the progress is shown below. Thank you for your interest in our work.

## For criteria 1, there's just yes/no and we simply filter and check them

## For criteria 2 - please check [readme for it](criteria2_answer_compatibility/readme.md)

### reference free:
- [x] [GPT-Score](criteria2_answer_compatibility/gpt-score-base/)
- [x] [GPT-Answer](criteria2_answer_compatibility/gpt-answer-base/)

## For criteria 3 - please check [readme for it](criteria3_giveness/readme.md)

### reference free:
- [x] [Rule-base](criteria3_giveness/rule-base)
- [x] [Hou (2021)](criteria3_giveness/Entity-classify)
- [x] [GPT-zero-shot](criteria3_giveness/gpt-base) 
- [x] [GPT-few-shot](criteria3_giveness/gpt-base) 


## For criteria 4 - please check [readme for it](criteria4_anchor_relevence/readme.md)

### reference free:
- [x] [Rule-base](criteria4_anchor_relevence/rule-base)
- [x] [GPT-Score](criteria4_anchor_relevence/gpt-score-base)
- [x] [GPT-zero-shot](criteria4_anchor_relevence/gpt-base) 
- [x] [GPT-few-shot](criteria4_anchor_relevence/gpt-base) 
- [x] [BLEU1-sim](criteria4_anchor_relevence/bleu1-smi)

## For reference based measures:
- [x] [all-reference-based](reference_based_measures)

