import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

output_filename = "C:\\dev\\KAI\\kai-951-design-of-autotags\\data\\output\\Bert\\meeting_proposed_results\\output_raw_scored.csv"
output_df = pd.read_csv(output_filename)

max_score = 10
min_score = -10

output_df["normalised_score"] = (output_df["score"] - min_score) / (max_score - min_score)
output_df["normalised_score"] = np.clip(output_df["normalised_score"], 0, 1)

questions_list = []

output_df["score_difference"] = 0
output_df["weighted_score"] = 0

for index, row in output_df.iterrows():
    if row["answer_ref"] == 0:
        subset_df = output_df[(output_df["question_ref"] == row["question_ref"]) & (output_df["conversation_id"] == row["conversation_id"])].reset_index(drop=True)
        average_score = subset_df["normalised_score"].mean()
        score_difference = (row["normalised_score"] - average_score) / average_score
        output_df.loc[index, "score_difference"] = score_difference
        output_df.loc[index, "weighted_score"] = row["normalised_score"] / (score_difference)

output_df.drop(output_df[output_df["answer_ref"] != 0].index, inplace=True)

answer_data = []

conversations_id_list = output_df["conversation_id"].unique()
for conversation_id in conversations_id_list:
    subset_df = output_df[output_df["conversation_id"] == conversation_id].reset_index(drop=True)
    answers = subset_df["answer"].values
    if np.all(answers == ["no_answer", "no_answer"]):
        answer_data.append({
            "conversation_id": conversation_id,
            "question": subset_df["question"].iloc[0],
            "answer": "no_answer",
            "label": subset_df["label"].iloc[0],
            "prediction": 1
        })
    else:
        if "no_answer" in answers:
            subset_df.drop(subset_df[subset_df["answer"] == "no_answer"].index, inplace=True)
        row = subset_df[subset_df["weighted_score"] == subset_df["weighted_score"].max()]
        if row["weighted_score"].iloc[0] < 2:
            prediction = 0
        else:
            prediction = 1
        answer_data.append({
            "conversation_id": conversation_id,
            "question": row["question"].iloc[0],
            "answer": row["answer"].iloc[0],
            "label": row["label"].iloc[0],
            "prediction": prediction
        })

answer_df = pd.DataFrame(answer_data)

accuracy = accuracy_score(answer_df["label"], answer_df["prediction"])
precision = precision_score(answer_df["label"], answer_df["prediction"])
recall = recall_score(answer_df["label"], answer_df["prediction"])
f1 = f1_score(answer_df["label"], answer_df["prediction"])

text_file = f"""Meeting Proposed Results:
accuracy: {accuracy}
precision: {precision}
recall: {recall}
f1: {f1}"""

with open("C:\\dev\\KAI\\kai-951-design-of-autotags\\data\\output\\Bert\\meeting_proposed_results\\scores.txt", "w") as f:
    f.write(text_file)

answer_df.to_csv("C:\\dev\\KAI\\kai-951-design-of-autotags\\data\\output\\Bert\\meeting_proposed_results\\results.csv", index=False)
