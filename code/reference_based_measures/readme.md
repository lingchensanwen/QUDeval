### This is the script to automatically evaluate reference based measures

### Before running you may want to set your openai keys for the gpt score

### Directory structure

example_reference_data.txt - example article, please replace this with articles you want to run eval on
example_reference_tuning.txt - example article, please replace this with instances that you want to use to tune the threshold

### [For calculating thresholds](https://github.com/lingchensanwen/QUDeval/tree/main/code/reference_based_measures)
<code>!python find_best_threshold</code> Run this on a small validation set to compute best thresholds for mapping the reference based to reference free criteria

### [For computing scores](https://github.com/lingchensanwen/QUDeval/tree/main/code/reference_based_measures)
<code>!python eval_by_threshold</code> This computes the reference based scores and calculates the F1 for the mapping to the reference free criteria based on the above computed thresholds

### Question Sensitive Text Similarity
We cloned [QSTS](https://github.com/NUS-IDS/coling22_QSTS) and changed the combining function to an arithmetic mean instead of a geometric mean 


