### This is the script to automatic evaluate criteria 4 (Relevance)

Note for this criteria, label 1 is well grounded, label 2 is part grounded, label 3 is not grounded

### before running you may wanna set your openai keys for gpt-base 

### Directory structure

example_reference_data.txt - example article, please replace this with articles you want to run eval on.
example_reference_tuning.txt - example article, please replace this with instances that you want to use to tune the threshold

### [for gpt base](https://github.com/lingchensanwen/QUDeval/tree/main/code/criteria4_anchor_relevence/gpt-base) 
<code>!python gpt_response</code> to get zero-shot and few-shot gpt eval, by default we're using the best model gpt-4.

### [for gpt score base](https://github.com/lingchensanwen/QUDeval/tree/main/code/criteria4_anchor_relevence/gpt-score-base)
To run this, you need to define a small set of your data to find best threshold on it, and then using this threshold to classify your main data.
In this example, we use it as whole set to compute just as an example. In your case, please define a val set and pick best threshold on it and run on test set.

To obatin best threshold run <code>!python find_best_threshold</code> and you will generate best threshold saved in best_threshold_info.txt

Then to classify your files on threshold run <code>!python eval_by_threshold</code>. 

### [for rule base](https://github.com/lingchensanwen/QUDeval/tree/main/code/criteria4_anchor_relevence/rule-base)
This depends on np base method as well. Please check [this path](https://github.com/lingchensanwen/QUDeval/tree/main/code/criteria3_giveness/rule-base) to genearte required np file.
Then run <code>!python eval_rule.py</code>

### [for BLEU1-smi](https://github.com/lingchensanwen/QUDeval/tree/main/code/criteria4_anchor_relevence/bleu1-smi) 
To obatin best threshold run <code>!python find_best_threshold</code> and you will generate best threshold saved in best_threshold_info.txt
Then to classify your files on threshold run <code>!python eval_by_threshold</code>. 
