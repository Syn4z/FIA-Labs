from parrot import Parrot
import torch
import json

# Initialize Parrot
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=torch.cuda.is_available())

def paraphrase_sentence(sentence):
    # Generate paraphrases
    paraphrases = parrot.augment(input_phrase=sentence)
    return paraphrases

# Load questions and answers from the JSON file
with open("similar_dataset.json", "r") as f:
    data = json.load(f)

entries = data['questions']

new_entries = []
for i, entry in enumerate(entries):
    question = entry["question"]
    answer = entry["answer"]

    print(f"\n Original question {i+1}: {question}")
    print("Original Answer:", answer)

    question_paraphrases = paraphrase_sentence(question)
    if question_paraphrases:
        for paraphrase in question_paraphrases:
            print(paraphrase[0])

    answer_paraphrases = paraphrase_sentence(answer)
    if answer_paraphrases:
        for paraphrase in answer_paraphrases:
            print(paraphrase[0])

    if question_paraphrases and answer_paraphrases:
        for q in question_paraphrases:
            for a in answer_paraphrases:
                new_entries.append({"question": q[0], "answer": a[0]})
    elif question_paraphrases:
        for q in question_paraphrases:
            new_entries.append({"question": q[0], "answer": answer})
    elif answer_paraphrases:
        for a in answer_paraphrases:
            new_entries.append({"question": question, "answer": a[0]})

# Save the new entries to a JSON file
with open("pharaphrased_dataset.json", "w") as f:
    json.dump({"questions": new_entries}, f, indent=4)
