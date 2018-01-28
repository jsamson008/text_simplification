import json
import sys

# from pycorenlp import StanfordCoreNLP

file_name = sys.argv[1]

# nlp = StanfordCoreNLP('http://localhost:9000')

print("Reading")
with open(file_name) as f:
    data = json.load(f)
print("Done")

bleu = 0
sari = 0
cnt = 0
i = 0
for item in data:
    print(str(i) + " / " + str(len(data)))
    i += 1
    candidates = []
    for k,v in item['candidates'].iteritems():
        idx = int(k)
        center = item['centers'][idx]
        if len(center) > 0:
            center = center[0]
        else:
            continue
        for c in v:
            if c['original_rank'] == center:
                candidates.append(c)
                break

    new_candidates = []
    for c in candidates:
        # sent = c['sent']
        # num_verbs = 0
        # num_adverbs = 0
        # num_adj = 0
        # num_nouns = 0
        # pos_sent = nlp.annotate(sent, properties={'annotators': 'tokenize,ssplit,pos','outputFormat': 'json', 'ner.model': 'edu/stanford/nlp/models/ner/english.conll.4class.distsim.crf.ser.gz'})
        # for sentence in pos_sent['sentences']:
        #     for token in sentence['tokens']:
        #         if token["pos"] in ["NN", "NNS", "NNP", "NNPS"]:
        #             num_nouns += 1
        #         elif token["pos"] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
        #             num_verbs += 1
        #         elif token["pos"] in ["JJ", "JJR", "JJS"]:
        #             num_adj += 1
        #         elif token["pos"] in ["RB", "RBR", "RBS"]:
        #             num_adverbs += 1
        # c['num_verbs'] = num_verbs
        # c['num_nouns'] = num_nouns
        # c['num_adj'] = num_adj
        # c['num_adverbs'] = num_adverbs
        new_candidates.append(c)

    candidates = sorted(new_candidates, key=lambda item: item['val']['lstm_lm'])
    for c in candidates[:1]:
        bleu += c['val']['bleu']
        sari += c['val']['sari']
        cnt += 1

print(float(bleu) / float(cnt))
print(float(sari) / float(cnt))
