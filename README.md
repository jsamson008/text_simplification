# text_simplification
Deep Learning based Text Simplification

# Steps

1.  python anonymize.py data/Zhang_Lapata_Newsela_splits/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test[.src/.dst]

2. python preprocess.py -train_src ../text_simplification/new_data/train/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.train.src.aner -train_tgt ../text_simplification/new_data/train/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.train.dst.aner -valid_src ../text_simplification/new_data/valid/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.valid.src.aner -valid_tgt ../text_simplification/new_data/valid/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.valid.dst.aner -save_data ../text_simplification/preprocess_data/data -share_vocab

3. python train.py -data ../text_simplification/preprocess_data/data -save_model ../text_simplification/model/model -batch_size 32 -epochs 20 -dropout 0.2 -optim adam -learning_rate 0.001 -report_every 1 -share_embeddings -global_attention general -encoder_type brnn -gpuid 0 -rnn_size 128 -layers 1

4. python translate.py -model ../text_simplification/model/model_acc_52.94_ppl_23.79_e5.pt -src ../text_simplification/new_data/test/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.src.aner -output ../text_simplification/new_data/test/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.predicted.beam400 -share_vocab -replace_unk -verbose -beam_size 400 -n_best 400 -gpu 1 -batch_size 32

5. python deanonymize.py data/new_data/test/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.deanonymiser data/new_data/test/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.src data/new_data/test/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.dst data/new_data/test/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.test.predicted.beam400 

<Now replace empty lines with special character in the dictionary created>

6. th preprocess.lua -data_type monotext -train ../text_simplification/new_data/monotext/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.train -valid ../text_simplification/new_data/monotext/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.valid -save_data ../text_simplification/preprocess_data/datalm

7. th train.lua -model_type lm -data ../text_simplification/preprocess_data/datalm-train.t7 -save_model ../text_simplification/model/model-lm

8. th lm.lua score -model ../lm_models/model_epoch5_205.88.t7 -src ../data/new_data/test/dictionary_seq2seq_att.txt >../data/new_data/test/dictionary_lm.txt

9. python build_dictionary.py data/new_data/test/dictionary_seq2seq_att.txt data/new_data/test/dictionary_lm.txt

10. python evalute.py

11. 