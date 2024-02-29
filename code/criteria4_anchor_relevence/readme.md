### This is the script to automatic evaluate criteria 4 (Relevance)

### before running you may wanna set your openai keys for gpt-base 

### [for gpt base](QUDeval/code/criteria4_anchor_relevence/gpt-base)
<code>!python gpt_response</code> to get zero-shot and few-shot gpt eval, by default we're using the best model gpt-4.

### [for gpt score base](/home/yw23374/QUDeval/code/criteria4_anchor_relevence/gpt-score-base)
To run this, you need to define a small set of your data to find best threshold on it, and then using this threshold to classify your main data.
In our paper, this is best threshold we found under small set is 20, 80. In this example, we use it as whole set to compute this so there's difference but we just run it to show an example.

To obatin best threshold run <code>!python find_best_threshold</code> and you will generate best threshold saved in best_threshold_info.txt

Then to classify your files on threshold run <code>!python eval_by_threshold</code>. This is very similiar to find_best_threshold. 

### [for rule base](QUDeval/code/criteria4_anchor_relevence/rule-base)
This depends on np base method as well. Please check [this path](QUDeval/code/criteria3_giveness) to genearte required np file.
<code>!python eval_rule.py</code>