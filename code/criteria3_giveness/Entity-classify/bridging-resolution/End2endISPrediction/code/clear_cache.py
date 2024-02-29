import os

file1 = "/home/yw23374/bridging-resolution/End2endISPrediction/entity_classify/data/cached_dev_output_256_is8"  # Replace with the actual path of file1
file2 = "/home/yw23374/bridging-resolution/End2endISPrediction/entity_classify/output/eval_prediction.txt"  # Replace with the actual path of file2
file3 = "/home/yw23374/bridging-resolution/End2endISPrediction/entity_classify/data/test.tsv"

if os.path.isfile(file1):
    os.remove(file1)
    print(f"File {file1} has been successfully removed.")
else:
    print(f"File {file1} does not exist.")

if os.path.isfile(file2):
    os.remove(file2)
    print(f"File {file2} has been successfully removed.")
else:
    print(f"File {file2} does not exist.")

if os.path.isfile(file3):
    os.remove(file3)
    print(f"File {file3} has been successfully removed.")
else:
    print(f"File {file3} does not exist.")