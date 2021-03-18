```
python extract_features.py \                                      
  --input_file=./test/embedding.txt \
  --output_file=./test/output.json \
  --vocab_file=./model/vocab.txt \
  --bert_config_file=./model/bert_config.json \
  --init_checkpoint=./model/20200621_Python_pre_trained_model__epochs_2__length_512_model.ckpt-602440 \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=512 \
  --batch_size=8
```