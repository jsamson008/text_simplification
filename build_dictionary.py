import Levenshtein as L
import sys
import json
from subprocess import Popen, PIPE
from nltk.translate.bleu_score import sentence_bleu
import sari

from datetime import datetime

dictionary_file = sys.argv[1]
language_model_file = sys.argv[2]

def calc_bleu(cand_smpl, gold_smpl):
    gold_smpl_list_tok = [gold_smpl.split(" ")]
    cand_smpl_tok = cand_smpl.split(" ")
    return sentence_bleu(gold_smpl_list_tok, cand_smpl_tok)

def calc_sari(cplx, cand_smpl, gold_smpl):
    return sari.SARIsent(cplx, cand_smpl, [gold_smpl])

print("[" + str(datetime.now()) + "]: Loading dictionary")
with open(dictionary_file) as f:
    data = f.readlines()
data = [l.strip() for l in data]

print("[" + str(datetime.now()) + "]: Loading LSTM scores")
with open(language_model_file) as f:
    nn_lm_scores = f.readlines()
nn_lm_scores = [l.strip().split(":")[-1].strip() for l in nn_lm_scores]

# print("[" + str(datetime.now()) + "]: Loading Berkeley scores")
# with open("dictionary_with_brnn_berkeleylm_logprob.txt") as f:
#     berkeley_lm_scores = f.readlines()
# berkeley_lm_scores = [l.strip() for l in berkeley_lm_scores]

# print("[" + str(datetime.now()) + "]: Loading autoenc distances")
# with open("dictionary_with_brnn_autoenc_distances.txt") as f:
#     autoenc_dist = f.readlines()
# autoenc_dist = [l.strip() for l in autoenc_dist]

# print("[" + str(datetime.now()) + "]: Loading text entailment probs")
# with open("dictionary_with_brnn_textual_entailment_scores.txt") as f:
#     text_entailment_score = f.readlines()
# text_entailment_score = [l.strip().split(" ") for l in text_entailment_score]

print("[" + str(datetime.now()) + "]: Building final dict")
idx = 0
parsed_data = []
while(True):
    if idx == len(data):
        break
    complex_sent = data[idx]
    idx += 1
    simple_sent = data[idx]
    idx += 1
    num_candidates = int(data[idx])
    candidates = []
    idx += 1
    cnt = 0
    while cnt < num_candidates:
        dist = L.distance(data[idx], simple_sent)
        print(str(idx) + " / " + str(len(data)))
        candidates.append({
            'sent': data[idx], 
            'edit_distance': dist, 
            # 'berkeley_lm': float(berkeley_lm_scores[idx]),
            'lstm_lm': float(nn_lm_scores[idx]),
            # 'autoenc': float(autoenc_dist[idx]),
            'bleu': calc_bleu(data[idx], simple_sent),
            'sari': calc_sari(complex_sent, data[idx], simple_sent),
            # 'text_entail': [float(l.strip()) for l in text_entailment_score[idx]]
        })
        idx += 1
        cnt += 1
    print("*********************")
    print(complex_sent)
    print(simple_sent)
    parsed_data.append({
        'complex': complex_sent,
        'gold': simple_sent,
        'candidates': candidates
    })

with open("data/new_data/test/dictionary_seq2seq_att.json", "w") as f:
    json.dump(parsed_data, f, sort_keys=True, indent=4)
