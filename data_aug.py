import csv
import os
import openai
from datasets import load_dataset
from tqdm import tqdm

cares = load_dataset("/mnt/beegfs/jcollado/hf-datasets/CARES")

cares = cares["train"]

# cares_to_augment = cares.select([0, 1, 2])

chapters_to_augment = [7, 9, 12, 17, 19]
cares_to_augment = cares.filter(lambda example: any(chapter in example["chapters"] for chapter in chapters_to_augment))

cares_to_augment = cares_to_augment.filter(lambda example, idx: idx > 267, with_indices=True)

openai.api_key = os.environ["OPENAI_API_KEY"]

def rewrite_report(report):
    prompt = "Reescribe el siguiente informe radiológico sin utilizar listas de puntos: " + report

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content

# Augment dataset
with open("/mnt/beegfs/jcollado/TFM/data/cares-augmented-7.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(["iddoc", "id", "full_text", "chapters"])
    for row in tqdm(cares_to_augment):
        new_report = rewrite_report(row["full_text"])
        writer.writerow([row["iddoc"], row["id"], new_report, row["chapters"]])