import json

with open('data/new_data/test/dictionary_seq2seq_att.json') as f:
    data = json.load(f)

bleu = 0
sari = 0
cnt = 0
i = 0
for item in data:
    print(str(i) + " / " + str(len(data)))
    i += 1
    for cand in item['candidates'][:1]:
        bleu += cand['bleu']
        sari += cand['sari']
        cnt += 1

print(float(bleu) / float(cnt))
print(float(sari) / float(cnt))
