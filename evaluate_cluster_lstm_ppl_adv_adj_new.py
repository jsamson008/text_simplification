import json
import sys

from pycorenlp import StanfordCoreNLP

file_name = sys.argv[1]

nlp = StanfordCoreNLP('http://localhost:9000')

with open(file_name) as f:
    data = json.load(f)

def filter_pos(sent):
    sent = str(sent)
    num_verbs = 0
    num_adverbs = 0
    num_adj = 0
    num_nouns = 0
    pos_sent = nlp.annotate(sent, properties={'annotators': 'tokenize,ssplit,pos','outputFormat': 'json', 'ner.model': 'edu/stanford/nlp/models/ner/english.conll.4class.distsim.crf.ser.gz'})
    for sentence in pos_sent['sentences']:
        for token in sentence['tokens']:
            if token["pos"] in ["NN", "NNS", "NNP", "NNPS"]:
                num_nouns += 1
            elif token["pos"] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
                num_verbs += 1
            elif token["pos"] in ["JJ", "JJR", "JJS"]:
                num_adj += 1
            elif token["pos"] in ["RB", "RBR", "RBS"]:
                num_adverbs += 1
    cnt = num_adj + num_adverbs
    return cnt

bleu = 0
sari = 0
cnt = 0
i = 0
for item in data:
    print(str(i) + " / " + str(len(data)))
    i += 1
    centers = item['centers']
    candidates = []
    for k,v in item['candidates'].iteritems():
        idx = int(k)
        center = centers[idx][0]
        temp = []
        for c in v:
            if c['original_rank'] == center:
                temp.append(c)
                break
        candidates.extend(temp)

    # new_candidates = []
    # for c in candidates:
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
        # new_candidates.append(c)

    c = min(candidates, key=lambda item: filter_pos(item['val']['sent']))
    bleu += c['val']['bleu']
    sari += c['val']['sari']
    cnt += 1

print(float(bleu) / float(cnt))
print(float(sari) / float(cnt))
