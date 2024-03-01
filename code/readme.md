
In this implementation, there's many approaches calling gpt so we save multiple files in the code to make the paying for api meaningful. For each criteria please go to corresponding directory and check its readme file. For your implemenation, you're expected to set up file path for desired inputs and outputs. We're still actively modifying and cleaning the code. And the progress is shown below. Thank you for your interest in our work.


## For criteria 1, there's just yes/no and we simply filter and check them

## For criteria 2

### reference free:
- [ ] GPT-Score
- [ ] GPT-Answer
### reference based:
- [ ] BLEU-1
- [ ] BERTScore
- [ ] METEOR
- [ ] ROUGE
- [ ] QSTS
- [ ] GPT-Score


## For criteria 3

### reference free:
- [x] [Rule-base](criteria3_giveness/rule-base)
- [x] [Hou (2021)](criteria3_giveness/Entity-classify)
- [x] [GPT-zero-shot](criteria3_giveness/gpt-base) 
- [x] [GPT-few-shot](criteria3_giveness/gpt-base) 

### reference based:
- [ ] BLEU-1
- [ ] BERTScore
- [ ] METEOR
- [ ] ROUGE
- [ ] QSTS
- [ ] GPT-Score


## For criteria 4

### reference free:
- [x] [Rule-base](criteria4_anchor_relevence/rule-base)
- [x] [GPT-Score](criteria4_anchor_relevence/gpt-score-base)
- [x] [GPT-zero-shot](criteria4_anchor_relevence/gpt-base) 
- [x] [GPT-few-shot](criteria4_anchor_relevence/gpt-base) 
- [ ] BLEU1-sim

### reference based:
- [ ] BLEU-1
- [ ] BERTScore
- [ ] METEOR
- [ ] ROUGE
- [ ] QSTS
- [ ] GPT-Score
