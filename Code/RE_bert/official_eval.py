import subprocess
import os
import re

def parse_metrics(result_text):
    # Find and extract the macro F1 Score from the result text
    macro_f1_match = re.search(r'Macro F1 Score: (\d+\.\d+)', result_text)
    if macro_f1_match:
        macro_f1_score = float(macro_f1_match.group(1))*100
        return macro_f1_score
    else:
        raise Exception("Macro F1 Score not found in the result.txt file")

def official_f1():
    cmd = 'python eval/scorer.py eval/proposed_answers.txt eval/answer_keys.txt > eval/result.txt'
    try:
        subprocess.run(cmd, shell=True, check=True)
        
        with open("eval/result.txt", "r", encoding="utf-8") as f:
            result_text = f.read()
        
        macro_f1_score = parse_metrics(result_text)
        return macro_f1_score
        
    except subprocess.CalledProcessError:
        raise Exception("Error while running scorer.py or files are missing")
    except Exception as e:
        raise Exception(f"Error parsing metrics: {e}")

if __name__ == "__main__":
    try:
        macro_f1_score = official_f1()
        formatted_macro_f1 = "{:.4f}".format(macro_f1_score)
        print("Macro-averaged F1 = {}%".format(formatted_macro_f1))
    except Exception as e:
        print(f"Error: {e}")
