import json
import sys
import traceback

deanonymiser_file = sys.argv[1]
complex_file = sys.argv[2]
gold_simple_file = sys.argv[3]
beam_file = sys.argv[4]

def deanonymize(sentence, deanonymizer):
    deanonymizer = json.loads(deanonymizer)
    deanonymized_sentence = []
    for token in sentence.split(" "):
            deanonymized_sentence.append(deanonymizer.get(token, token).decode("utf-8"))
    return " ".join(deanonymized_sentence)

with open(deanonymiser_file) as f:
    deanonymizer = f.readlines()
deanonymizer = [l.strip() for l in deanonymizer]

with open(complex_file) as f:
    gold_complex_data = f.readlines()
gold_complex_data = [l.strip() for l in gold_complex_data]

with open(gold_simple_file) as f:
    gold_simple_data = f.readlines()
gold_simple_data = [l.strip() for l in gold_simple_data]

# Candidate file w/ beam output
with open(beam_file) as f:
    predicted_simple_data = f.readlines()
predicted_simple_data = [l.strip() for l in predicted_simple_data]

beam_size = 400

with open("data/new_data/test/dictionary_seq2seq_att.txt", "w") as f:
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
