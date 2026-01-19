import csv
from datetime import datetime
import os
import json
def read_word_pairs(csv_file):
    word_pairs = []
    with open(csv_file,mode="r",encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            word_pairs.append((row[0],row[1]))
    
    return word_pairs

def log_with_timestamp(log,message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.append(f"[{timestamp}] {message}")



def save_log_to_file(log, file_path="game_log.txt"):
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding='utf-8') as f:
            for entry in log:
                f.write(entry + "\n")
    else:
        with open(file_path, "a", encoding='utf-8') as f:
            for entry in log:
                f.write(entry + "\n")

def read_json(json_path):
    with open(json_path,"r") as f:
        return json.load(f)