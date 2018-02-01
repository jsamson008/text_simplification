import sys
from pycorenlp import StanfordCoreNLP
import json

def process(complex_sent, simple_sent, nlp):
    complex_sent = complex_sent.decode("ascii", "ignore").encode("ascii", "ignore")
    simple_sent = simple_sent.decode("ascii", "ignore").encode("ascii", "ignore")
    print("*********************************")
    print(complex_sent)
    print(simple_sent)
    complex_annotated = nlp.annotate(complex_sent, properties={'annotators': 'tokenize,ssplit,pos,ner','outputFormat': 'json', 'ner.model': 'edu/stanford/nlp/models/ner/english.conll.4class.distsim.crf.ser.gz'})
    simple_annotated = nlp.annotate(simple_sent, properties={'annotators': 'tokenize,ssplit,pos,ner','outputFormat': 'json', 'ner.model': 'edu/stanford/nlp/models/ner/english.conll.4class.distsim.crf.ser.gz'})
    complex_tagged = []
    simple_tagged = []
    for sent in complex_annotated['sentences']:
        for token in sent['tokens']:
            token['ner'] = "O" if (token['ner'] != "PERSON" and token['ner'] != "ORGANIZATION" and token['ner'] != "LOCATION") else token['ner']
            complex_tagged.append({"word": token['word'].lower(), "ner": token['ner']})
    for sent in simple_annotated['sentences']:
        for token in sent['tokens']:
            token['ner'] = "O" if (token['ner'] != "PERSON" and token['ner'] != "ORGANIZATION" and token['ner'] != "LOCATION") else token['ner']
            simple_tagged.append({"word": token['word'].lower(), "ner": token['ner']})
    complex_sent = []
    current_words = []
    current_type = ""
    cnt = {}
    for token in complex_tagged:
        if current_type != "" and current_type != token["ner"]:
            if current_type not in cnt:
                cnt[current_type] = 0
            cnt[current_type] += 1
            entity = " ".join([w['word'] for w in current_words])
            complex_sent.append({"word": entity, "ner": current_type, "cnt": cnt[current_type]})
            current_type = ""
            current_words = []
        if token['ner'] == "O":
            complex_sent.append(token)
        else:
            current_words.append(token)
            current_type = token['ner']
    simple_sent = []
    current_words = []
    current_type = ""
    for token in simple_tagged:
        if current_type != "" and current_type != token["ner"]:
            if current_type not in cnt:
                cnt[current_type] = 0
            cnt[current_type] += 1
            entity = " ".join([w['word'] for w in current_words])
            simple_sent.append({"word": entity, "ner": current_type, "cnt": cnt[current_type]})
            current_type = ""
            current_words = []
        if token['ner'] == "O":
            simple_sent.append(token)
        else:
            current_words.append(token)
            current_type = token['ner']
    entity_mapping = {}
    for token in complex_sent:
        if token['ner'] != "O":
            if token['ner'] not in entity_mapping:
                entity_mapping[token['ner']] = {}
            if token['word'] not in entity_mapping[token['ner']]:
                entity_mapping[token['ner']][token['word']] = token['cnt']
    for token in simple_sent:
        if token['ner'] != "O":
            if token['ner'] not in entity_mapping:
                entity_mapping[token['ner']] = {}
            if token['word'] not in entity_mapping[token['ner']]:
                entity_mapping[token['ner']][token['word']] = token['cnt']
    c = []
    s = []
    deanonymize = {}
    for token in complex_sent:
        if token["ner"] == "O":
            c.append(token["word"])
        else:
            c.append(token["ner"] + "@" + str(entity_mapping[token["ner"]][token["word"]]))
            deanonymize[token["ner"] + "@" + str(entity_mapping[token["ner"]][token["word"]])] = token["word"]
    for token in simple_sent:
        if token["ner"] == "O":
            s.append(token["word"])
        else:
            s.append(token["ner"] + "@" + str(entity_mapping[token["ner"]][token["word"]]))
            deanonymize[token["ner"] + "@" + str(entity_mapping[token["ner"]][token["word"]])] = token["word"]
    c = " ".join(c)
    s = " ".join(s)
    return c,s,deanonymize

def anonymize(file_prefix):
    
    '''
        Start a server using command: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
    '''
    complex_file_name = file_prefix + ".complex"
    simple_file_name = file_prefix + ".simple"
    nlp = StanfordCoreNLP('http://localhost:9000')

    complex_lines = []
    with open(complex_file_name) as f:
        complex_lines = f.readlines()
    complex_lines = [l.strip() for l in complex_lines]

    simple_lines = []
    with open(simple_file_name) as f:
        simple_lines = f.readlines()
    simple_lines = [l.strip() for l in simple_lines]

    pair = []
    cnt = 0
    for c,s in zip(complex_lines, simple_lines):
        cnt += 1
        print(str(cnt) + " / " + str(len(complex_lines)))
        post_c, post_s, deanonymize = process(c, s, nlp)
        pair.append((post_c, post_s, deanonymize))
    
    with open(complex_file_name + ".aner", "w") as f:
        for (c,_,_) in pair:
            f.write(c.encode())
            f.write("\n")
    with open(simple_file_name + ".aner", "w") as f:
        for (_,s,_) in pair:
            f.write(s.encode())
            f.write("\n")
    with open(data_type + ".deanonymiser", "w") as f:
        for (_,_,d) in pair:
            f.write(json.dumps(d))
            f.write("\n")

if __name__ == "__main__":
    file_prefix = sys.argv[1]
    anonymize(file_prefix)
