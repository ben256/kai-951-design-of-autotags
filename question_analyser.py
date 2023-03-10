import numpy as np
import pandas as pd

output_filename = "C:\\dev\\KAI\\kai-951-design-of-autotags\\data\\output\\Bert\\meeting_proposed_results\\output.csv"
output_df = pd.read_csv(output_filename)

max_score = 10
min_score = -10

output_df["normalised_score"] = (output_df["score"] - min_score) / (max_score - min_score)
output_df["normalised_score"] = np.clip(output_df["normalised_score"], 0, 1)
output_df["manual_label"].fillna(0, inplace=True)
output_df["weighted_score"] = output_df["manual_label"] * output_df["normalised_score"]

for answer in ["all", "no_answer_removed"]:

    questions_list = []

    if answer == "no_answer_removed":
        # remove rows where answer is "no_answer" without affecting output_df
        answer_df = output_df.drop(output_df[output_df["answer"] == "no_answer"].index)
    else:
        answer_df = output_df

    answer_df["score_difference"] = 0

    for index, row in answer_df.iterrows():
        if row["manual_label"] != 0:
            subset_df = answer_df[(answer_df["question_ref"] == row["question_ref"]) & (answer_df["conversation_id"] == row["conversation_id"])].reset_index(drop=True)
            average_score = subset_df["weighted_score"].mean()
            answer_df.loc[index, "score_difference"] = row["weighted_score"] - average_score

    question_ref_list = answer_df["question_ref"].unique()
    for question_ref in question_ref_list:
        question_df = answer_df[answer_df["question_ref"] == question_ref].reset_index(drop=True)
        non_zero_labels = [x for x in question_df["manual_label"] if x != 0]
        non_zero_scores = [x for x in question_df["weighted_score"] if x != 0]
        non_zero_differences = [x for x in question_df["score_difference"] if x != 0]

        average_label = sum(non_zero_labels) / (len(non_zero_labels) or 1)
        average_score = sum(non_zero_scores) / (len(non_zero_scores) or 1)
        average_difference = sum(non_zero_differences) / (len(non_zero_differences) or 1)

        questions_list.append({
            "question_ref": question_ref,
            "question": question_df["question"].iloc[0],
            "average_label": average_label,
            "average_score": average_score,
            "average_difference": average_difference
        })

    questions_df = pd.DataFrame(questions_list)
    questions_df.sort_values(by="average_difference", ascending=False, inplace=True)
    questions_df.to_csv(f"C:\\dev\\KAI\\kai-951-design-of-autotags\\data\\output\\Bert\\meeting_proposed_results\\results_{answer}.csv", index=False)
