import json
import sys
import traceback

def deanonymize(sentence, deanonymizer):
    deanonymizer = json.loads(deanonymizer)
    deanonymized_sentence = []
    for token in sentence.split(" "):
            deanonymized_sentence.append(deanonymizer.get(token, token).decode("utf-8"))
    return " ".join(deanonymized_sentence)

with open("test.deanonymiser") as f:
    deanonymizer = f.readlines()
deanonymizer = [l.strip() for l in deanonymizer]

with open("test.complex.aner") as f:
    gold_complex_data = f.readlines()
gold_complex_data = [l.strip() for l in gold_complex_data]

with open("test.simple.aner") as f:
    gold_simple_data = f.readlines()
gold_simple_data = [l.strip() for l in gold_simple_data]

# Candidate file w/ beam output
with open("test.complex.aner.simplified") as f:
    predicted_simple_data = f.readlines()
predicted_simple_data = [l.strip() for l in predicted_simple_data]

beam_size = 400

with open("dictionary.txt", "w") as f:
    for idx in range(len(gold_complex_data)):
        gold_complex = deanonymize(gold_complex_data[idx], deanonymizer[idx]).encode("utf-8")
        gold_simple = deanonymize(gold_simple_data[idx], deanonymizer[idx]).encode("utf-8")
        predicted_simple = predicted_simple_data[idx*400 : (idx+1)*400]
        predicted_simple = [deanonymize(l, deanonymizer[idx]).encode("utf-8") for l in predicted_simple]
        f.write(gold_complex)
        f.write("\n")
        f.write(gold_simple)
        f.write("\n")
        f.write(str(len(predicted_simple)))
        f.write("\n")
        f.write("\n".join(predicted_simple))
        f.write("\n")
