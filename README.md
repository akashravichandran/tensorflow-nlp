		Code has been run on Google Colab, thanks Google for providing computational resources

#### Contents

* Natural Language Processing（自然语言处理）

	* [Text Classification（文本分类）](https://github.com/zhedongzheng/finch#text-classification)
	
		* IMDB（English Data）

	* [Text Matching（文本匹配）](https://github.com/zhedongzheng/finch#text-matching)
	
		* SNLI（English Data）
	
		* 微众银行智能客服（Chinese Data）

	* [Spoken Language Understanding（对话理解）](https://github.com/zhedongzheng/finch#spoken-language-understanding)

		* ATIS（English Data）

	* [Generative Dialog（生成式对话）](https://github.com/zhedongzheng/finch#generative-dialog)

		* 青云语料（Chinese Data）
				
			* Python Inference（基于 Python 的推理）
					
			* Java Inference（基于 Java 的推理）
		
	* [Multi-turn Dialogue Rewriting（多轮对话改写）](https://github.com/zhedongzheng/finch#multi-turn-dialogue-rewriting)

		* 20k 腾讯 AI 研发数据（Chinese Data）
				
			* Python Inference（基于 Python 的推理）

			* Java Inference（基于 Java 的推理）

	* [Semantic Parsing（语义解析）](https://github.com/zhedongzheng/finch#semantic-parsing)
	
		* Facebook AI Research Data（English Data）
	
	* [Question Answering（问题回答）](https://github.com/zhedongzheng/finch#question-answering)
	
		* bAbI（Engish Data）
		
	* [Text Processing Tools（文本处理工具）](https://github.com/zhedongzheng/finch#text-processing-tools)

* Knowledge Graph（知识图谱）

	* [Knowledge Graph Inference（知识图谱推理）](https://github.com/zhedongzheng/finch#knowledge-graph-inference)
	
	* [Knowledge Base Question Answering（知识图谱问答）](https://github.com/zhedongzheng/finch#knowledge-base-question-answering)
	
	* [Knowledge Graph Tools（知识图谱工具）](https://github.com/zhedongzheng/finch#knowledge-graph-tools)

* [Recommender System（推荐系统）](https://github.com/zhedongzheng/finch#recommender-system)

	* Movielens 1M（English Data）

---

## Text Classification

```
└── finch/tensorflow2/text_classification/imdb
	│
	├── data
	│   └── glove.840B.300d.txt          # pretrained embedding, download and put here
	│   └── make_data.ipynb              # step 1. make data and vocab: train.txt, test.txt, word.txt
	│   └── train.txt  		     # incomplete sample, format <label, text> separated by \t 
	│   └── test.txt   		     # incomplete sample, format <label, text> separated by \t
	│   └── train_bt_part1.txt  	     # (back-translated) incomplete sample, format <label, text> separated by \t
	│
	├── vocab
	│   └── word.txt                     # incomplete sample, list of words in vocabulary
	│	
	└── main              
		└── attention_linear.ipynb   # step 2: train and evaluate model
		└── attention_conv.ipynb     # step 2: train and evaluate model
		└── fasttext_unigram.ipynb   # step 2: train and evaluate model
		└── fasttext_bigram.ipynb    # step 2: train and evaluate model
		└── sliced_rnn.ipynb         # step 2: train and evaluate model
		└── sliced_rnn_bt.ipynb      # step 2: train and evaluate model
```

* Task: [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/)（English Data）
	
        Training Data: 25000, Testing Data: 25000, Labels: 2
	
	* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/data/make_data.ipynb)
		
		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/data/train.txt)
		
		* [\<Text File>: Data Example (Back-Translated)](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/data/train_bt_part1.txt)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/vocab/word.txt)

	* Model: TF-IDF + Logistic Regression
	
		* PySpark
		
			* [\<Notebook> Unigram + TF + IDF + Logistic Regression](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/spark/text_classification/imdb/tfidf_lr.ipynb)
			
			 	-> 88.2% Testing Accuracy
			
		* Sklearn
		
			* [\<Notebook> Unigram + TF + IDF + Logistic Regression](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/sklearn/text_classification/imdb/tfidf_lr_binary_false.ipynb)
			
			 	-> 88.3% Testing Accuracy
			
			* [\<Notebook> Unigram + TF (binary) + IDF + Logistic Regression](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/sklearn/text_classification/imdb/tfidf_lr_binary_true.ipynb)
			
			 	-> 88.8% Testing Accuracy

			* [\<Notebook> Unigram + Bigram + TF (binary) + IDF + Logistic Regression](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/sklearn/text_classification/imdb/tfidf_lr_binary_true_bigram.ipynb)
			
			 	-> 89.6% Testing Accuracy

	* Model: [FastText](https://arxiv.org/abs/1607.01759)
	
		* [Facebook Official Release](https://github.com/facebookresearch/fastText)
		
			* [\<Notebook> Unigram FastText](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/framework/official_fasttext/text_classification/imdb/unigram.ipynb)
		
		 		-> 87.3% Testing Accuracy
		
			* [\<Notebook> (Unigram + Bigram) FastText](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/framework/official_fasttext/text_classification/imdb/bigram.ipynb)

				-> 89.8% Testing Accuracy

			* [\<Notebook> Auto-tune FastText](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/framework/official_fasttext/text_classification/imdb/autotune.ipynb)

				-> 90.1% Testing Accuracy

		* TensorFlow 2

			* [\<Notebook> Unigram FastText](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/fasttext_unigram.ipynb)
				
			 	-> 89.1 % Testing Accuracy
				
			* [\<Notebook> (Unigram + Bigram) FastText](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/fasttext_bigram.ipynb)
	
	 			-> 90.2 % Testing Accuracy
	
	* Model: [Feedforward Attention](https://arxiv.org/abs/1512.08756)

		* TensorFlow 2

			* [\<Notebook> Feedforward Attention](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/attention_linear.ipynb)
			
			 	-> 89.5 % Testing Accuracy
			
			* [\<Notebook> CNN + Feedforward Attention](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/attention_conv.ipynb)

				-> 90.7 % Testing Accuracy

			* [\<Notebook> CNN + Feedforward Attention + Back-Translation + Char Embedding + Label Smoothing](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/cnn_attention_bt_char_label_smooth_cyclical.ipynb)
			
				-> 91.7 % Testing Accuracy

	* Model: [Sliced RNN](https://arxiv.org/abs/1807.02291)

		* TensorFlow 2

			* [\<Notebook> Sliced LSTM](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/sliced_rnn.ipynb)

 				-> 91.4 % Testing Accuracy

			* [\<Notebook> Sliced LSTM + Back-Translation](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/sliced_rnn_bt.ipynb)

 				-> 91.7 % Testing Accuracy
				
				```
				Back-Translation increases training data from 25000 to 50000

				which is done by "english -> french -> english" translation
				```

				```python
				from googletrans import Translator

				translator = Translator()

				translated = translator.translate(text, src='en', dest='fr').text
				
      			back = translator.translate(translated, src='fr', dest='en').text
				```

			* [\<Notebook> Sliced LSTM + Back-Translation + Char Embedding](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/sliced_rnn_bt_char.ipynb)

 				-> 92.3 % Testing Accuracy

			* [\<Notebook> Sliced LSTM + Back-Translation + Char Embedding + Label Smoothing](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/sliced_rnn_bt_char_label_smooth.ipynb)
			
				-> 92.5 % Testing Accuracy

			* [\<Notebook> Sliced LSTM + Back-Translation + Char Embedding + Label Smoothing + Cyclical LR](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_classification/imdb/main/sliced_rnn_bt_char_label_smooth_clr.ipynb)
			
				-> 92.6 % Testing Accuracy

				This result (without transfer learning) is higher than [CoVe](https://arxiv.org/pdf/1708.00107.pdf) (with transfer learning)

---

## Text Matching

```
└── finch/tensorflow2/text_matching/snli
	│
	├── data
	│   └── glove.840B.300d.txt       # pretrained embedding, download and put here
	│   └── download_data.ipynb       # step 1. run this to download snli dataset
	│   └── make_data.ipynb           # step 2. run this to generate train.txt, test.txt, word.txt 
	│   └── train.txt  		  # incomplete sample, format <label, text1, text2> separated by \t 
	│   └── test.txt   		  # incomplete sample, format <label, text1, text2> separated by \t
	│
	├── vocab
	│   └── word.txt                  # incomplete sample, list of words in vocabulary
	│	
	└── main              
		└── dam.ipynb      	  # step 3. train and evaluate model
		└── esim.ipynb      	  # step 3. train and evaluate model
		└── ......
```

* Task: [SNLI](https://nlp.stanford.edu/projects/snli/)（English Data）

        Training Data: 550152, Testing Data: 10000, Labels: 3

	* [\<Notebook>: Download Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/data/download_data.ipynb)
	
	* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/data/make_data.ipynb)
		
		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/data/train.txt)
		
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/vocab/word.txt)

	* TensorFlow 2

		* Model: [DAM](https://arxiv.org/abs/1606.01933)
		
			* [\<Notebook> DAM](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/dam.ipynb)
			
 				-> 85.3% Testing Accuracy
			
			 	The accuracy of this implementation is higher than [UCL MR Group](http://isabelleaugenstein.github.io/papers/JTR_ACL_demo_paper.pdf)'s implementation (84.6%)

		* Model: [Match Pyramid](https://arxiv.org/abs/1602.06359)
			
			* [\<Notebook> Pyramid](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/pyramid_multi_attn.ipynb)

 				-> 87.1% Testing Accuracy

	 		 	The accuracy of this model is 0.3% below ESIM, however the speed is 1x faster than ESIM

		* Model: [ESIM](https://arxiv.org/abs/1609.06038)
		
			* [\<Notebook> ESIM](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/esim.ipynb)

 				-> 87.4% Testing Accuracy

			 	The accuracy of this implementation is comparable to [UCL MR Group](http://isabelleaugenstein.github.io/papers/JTR_ACL_demo_paper.pdf)'s implementation (87.2%)

		* Model: [RE2](https://arxiv.org/abs/1908.00300)
		
			* [\<Notebook> RE2](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/re2_birnn.ipynb)

 				-> 87.7% Testing Accuracy

			* [\<Notebook> RE3](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/re2_3_birnn.ipynb)

 				-> 88.0% Testing Accuracy

			* [\<Notebook> RE3 + Cyclical LR + Label Smoothing](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/re2_3_birnn_label_smooth.ipynb)

				-> 88.3% Testing Accuracy

<img src="https://pic2.zhimg.com/80/v2-3ae885000f570573020afa0c4ce65a19_720w.jpg" height="300">

```
└── finch/tensorflow2/text_matching/chinese
	│
	├── data
	│   └── make_data.ipynb           # step 1. run this to generate char.txt and char.npy
	│   └── train.csv  		  # incomplete sample, format <text1, text2, label> separated by comma 
	│   └── test.csv   		  # incomplete sample, format <text1, text2, label> separated by comma
	│
	├── vocab
	│   └── cc.zh.300.vec             # pretrained embedding, download and put here
	│   └── char.txt                  # incomplete sample, list of chinese characters
	│   └── char.npy                  # saved pretrained embedding matrix for this task
	│	
	└── main              
		└── pyramid.ipynb      	  # step 2. train and evaluate model
		└── esim.ipynb      	  # step 2. train and evaluate model
		└── ......
```

* Task: [微众银行智能客服](https://github.com/terrifyzhao/text_matching/tree/master/input)（Chinese Data）

        Training Data: 100000, Testing Data: 10000, Labels: 2

	* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/chinese/data/make_data.ipynb)
		
		* [\<Text File>: Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/data/train.csv)
		
		* [\<Text File>: Vocabulary](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/vocab/char.txt)
		
	* Model	
	
		* TensorFlow 2
	
			* [\<Notebook> ESIM](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/chinese/main/esim.ipynb)

 				-> 82.5% Testing Accuracy

			* [\<Notebook> Pyramid](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/chinese/main/pyramid.ipynb)

 				-> 82.7% Testing Accuracy

			* [\<Notebook> RE2](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/chinese/main/re2.ipynb)

 				-> 82.9% Testing Accuracy

			* [\<Notebook> RE2 + Cyclical LR + Label Smoothing](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/text_matching/chinese/main/re2_cyclical_label_smooth.ipynb)

 				-> 83.8% Testing Accuracy

			These results are higher than [the results here](https://github.com/terrifyzhao/text_matching) and [the result here](https://github.com/liuhuanyong/SiameseSentenceSimilarity)

		* TensorFlow 1 + [bert4keras](https://github.com/bojone/bert4keras)
		
			* [\<Notebook> BERT](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/text_matching/chinese/main/bert_chinese_wwm.ipynb)
			
			 	-> 84.75% Testing Accuracy
			
				Weights downloaded from [here](https://github.com/ymcui/Chinese-BERT-wwm)

---

## Spoken Language Understanding

```
└── finch/tensorflow2/spoken_language_understanding/atis
	│
	├── data
	│   └── glove.840B.300d.txt           # pretrained embedding, download and put here
	│   └── make_data.ipynb               # step 1. run this to generate vocab: word.txt, intent.txt, slot.txt 
	│   └── atis.train.w-intent.iob       # incomplete sample, format <text, slot, intent>
	│   └── atis.test.w-intent.iob        # incomplete sample, format <text, slot, intent>
	│
	├── vocab
	│   └── word.txt                      # list of words in vocabulary
	│   └── intent.txt                    # list of intents in vocabulary
	│   └── slot.txt                      # list of slots in vocabulary
	│	
	└── main              
		└── bigru.ipynb               # step 2. train and evaluate model
		└── bigru_self_attn.ipynb     # step 2. train and evaluate model
		└── transformer.ipynb         # step 2. train and evaluate model
		└── transformer_elu.ipynb     # step 2. train and evaluate model
```

* Task: [ATIS](https://github.com/yvchen/JointSLU/tree/master/data)（English Data） 

	<img src="https://www.csie.ntu.edu.tw/~yvchen/f105-adl/images/atis.png" width="500">

        Training Data: 4978, Testing Data: 893

	* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/data/atis.train.w-intent.iob)

	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/data/make_data.ipynb)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/vocab/word.txt)

	* Model: [Bi-directional RNN](https://www.ijcai.org/Proceedings/16/Papers/425.pdf)
	
		* TensorFlow 2

			* [\<Notebook> Bi-GRU](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/bigru_clr.ipynb) 
			
			  97.4% Intent Acc, 95.4% Slot Micro-F1 on Testing Data

			* [\<Notebook> Bi-GRU + Self-Attention](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/bigru_self_attn_clr.ipynb) 
			
			  97.6% Intent Acc, 95.7% Slot Micro-F1 on Testing Data

			* [\<Notebook> Bi-GRU + CRF](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/bigru_clr_crf.ipynb) 
			
			  97.2% Intent Acc, 95.8% Slot Micro-F1 on Testing Data

	* Model: [ELMO Embedding](https://arxiv.org/abs/1802.05365)
	
		* TensorFlow 1

			* [\<Notebook> ELMO + Bi-GRU](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/spoken_language_understanding/atis/main/elmo_o1_bigru.ipynb) 
			
			  97.5% Intent Acc, 96.1% Slot Micro-F1 on Testing Data

			* [\<Notebook> ELMO + Bi-GRU + CRF](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/spoken_language_understanding/atis/main/elmo_o1_bigru_crf.ipynb) 
			
			  97.3% Intent Acc, 96.3% Slot Micro-F1 on Testing Data

---

## Generative Dialog

```
└── finch/tensorflow1/free_chat/chinese_qingyun
	│
	├── data
	│   └── raw_data.csv           		# raw data downloaded from external
	│   └── make_data.ipynb           	# step 1. run this to generate vocab {char.txt} and data {train.txt & test.txt}
	│   └── train.txt           		# processed text file generated by {make_data.ipynb}
	│
	├── vocab
	│   └── char.txt                	# list of chars in vocabulary for chinese
	│   └── cc.zh.300.vec			# fastText pretrained embedding downloaded from external
	│   └── char.npy			# chinese characters and their embedding values (300 dim)	
	│	
	└── main
		└── lstm_seq2seq_train.ipynb    # step 2. train and evaluate model
		└── lstm_seq2seq_export.ipynb   # step 3. export model
		└── lstm_seq2seq_infer.ipynb    # step 4. model inference
		└── transformer_train.ipynb     # step 2. train and evaluate model
		└── transformer_export.ipynb    # step 3. export model
		└── transformer_infer.ipynb     # step 4. model inference
```

* Task: 青云语料（Chinese Data）

        Training Data: 107687, Testing Data: 3350
	
	* Data

		* [\<Web Link>: Obtain Data](https://github.com/codemayq/chinese_chatbot_corpus)
		
		* [\<Text File>: Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_qingyun/data/train.txt)

		* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/free_chat/chinese_qingyun/data/make_data.ipynb)

	* Model: [RNN Seq2Seq + Attention](https://arxiv.org/abs/1409.0473)

		* TensorFlow 1
			
			* [\<Notebook> Training](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_qingyun/main/lstm_seq2seq_train.ipynb)
			
				LSTM + Attention + Beam Search -> 3.540 Testing Perplexity

		* Model Inference
		
			* [\<Notebook> Model Export](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_qingyun/main/lstm_seq2seq_export.ipynb)
			
			* [\<Notebook> Python Inference](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_qingyun/main/lstm_seq2seq_infer.ipynb)

	* Model: [Transformer](https://arxiv.org/abs/1706.03762)

		* TensorFlow 1 + [texar](https://github.com/asyml/texar)
			
			* [\<Notebook> Training](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_qingyun/main/transformer_train.ipynb)
			
				Transformer (6 Layers, 8 Heads) -> 3.540 Testing Perplexity

		* Model Inference
		
			* [\<Notebook> Model Export](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_qingyun/main/transformer_export.ipynb)
			
			* [\<Notebook> Python Inference](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/free_chat/chinese_qingyun/main/transformer_infer.ipynb)

```
└── FreeChatInference
	│
	├── data
	│   └── transformer_export/
	│   └── char.txt
	│   └── libtensorflow-1.14.0.jar
	│   └── tensorflow_jni.dll
	│
	└── src              
	    └── ModelInference.java
```

* [\<Notebook> Java Inference](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/java/FreeChatInference/src/ModelInference.java)

---

## Semantic Parsing

<img src="https://pic3.zhimg.com/v2-fa2cdccee8c725af42564b37741ba47a_b.jpg">

```
└── finch/tensorflow2/semantic_parsing/tree_slu
	│
	├── data
	│   └── glove.840B.300d.txt     	# pretrained embedding, download and put here
	│   └── make_data.ipynb           	# step 1. run this to generate vocab: word.txt, intent.txt, slot.txt 
	│   └── train.tsv   		  	# incomplete sample, format <text, tokenized_text, tree>
	│   └── test.tsv    		  	# incomplete sample, format <text, tokenized_text, tree>
	│
	├── vocab
	│   └── source.txt                	# list of words in vocabulary for source (of seq2seq)
	│   └── target.txt                	# list of words in vocabulary for target (of seq2seq)
	│	
	└── main
		└── lstm_seq2seq_tf_addons.ipynb           # step 2. train and evaluate model
		└── ......
		
```

* Task: [Semantic Parsing for Task Oriented Dialog](https://aclweb.org/anthology/D18-1300)（English Data）

        Training Data: 31279, Testing Data: 9042

	* [\<Text File>: Data Example](https://github.com/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/data/train.tsv)

	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/data/make_data.ipynb)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/vocab/target.txt)

	* Model: [RNN Seq2Seq + Attention](https://arxiv.org/abs/1409.0473)

		* TensorFlow 2

			* [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/gru_seq2seq_tf_addons_clr.ipynb) GRU + Attention + Beam Search + Cyclical LR + Label Smoothing ->
			
			  74.1% Exact Match on Testing Data

			* [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/lstm_seq2seq_tf_addons_clr.ipynb) LSTM + Attention + Beam Search + Cyclical LR + Label Smoothing ->
			
			  74.1% Exact Match on Testing Data

---

## Knowledge Graph Inference

```
└── finch/tensorflow2/knowledge_graph_completion/wn18
	│
	├── data
	│   └── download_data.ipynb       	# step 1. run this to download wn18 dataset
	│   └── make_data.ipynb           	# step 2. run this to generate vocabulary: entity.txt, relation.txt
	│   └── wn18  		          	# wn18 folder (will be auto created by download_data.ipynb)
	│   	└── train.txt  		  	# incomplete sample, format <entity1, relation, entity2> separated by \t
	│   	└── valid.txt  		  	# incomplete sample, format <entity1, relation, entity2> separated by \t 
	│   	└── test.txt   		  	# incomplete sample, format <entity1, relation, entity2> separated by \t
	│
	├── vocab
	│   └── entity.txt                  	# incomplete sample, list of entities in vocabulary
	│   └── relation.txt                	# incomplete sample, list of relations in vocabulary
	│	
	└── main              
		└── distmult_1-N.ipynb    	# step 3. train and evaluate model
```

* Task: WN18

        Training Data: 141442, Testing Data: 5000

	* [\<Notebook>: Download Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/download_data.ipynb)
	
		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/wn18/train.txt)
	
	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/make_data.ipynb)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/vocab/relation.txt)
	
	* We use [1-N Fast Evaluation](https://arxiv.org/abs/1707.01476) to largely accelerate evaluation process

		 <img src="https://pic4.zhimg.com/80/v2-8cd8481856f101af45501078b04456bb_720w.jpg">

	* Model: [DistMult](https://arxiv.org/abs/1412.6575)

		* TensorFlow 2

			* [\<Notebook> DistMult -> 79.7% MRR on Testing Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/main/distmult_1-N_clr.ipynb)
	
	* Model: [TuckER](https://arxiv.org/abs/1901.09590)
	
		* TensorFlow 2
		
			* [\<Notebook> TuckER -> 88.5% MRR on Testing Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/main/tucker_1-N_clr.ipynb)
	
	* Model: [ComplEx](https://arxiv.org/abs/1606.06357)

		* TensorFlow 2

			* [\<Notebook> ComplEx -> 93.8% MRR on Testing Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/main/complex_1-N_clr.ipynb)

---

## Knowledge Graph Tools

* Data Scraping

	* [Using Scrapy](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/scrapy/car.ipynb)

	* [Downloaded](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/scrapy/car.csv)

* SPARQL

	* [WN18 Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/rdf_sparql_test.ipynb)

* Neo4j + Cypher

	* [Getting Started](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/neo4j/install_neo4j.ipynb)

---

## Question Answering

<img src="https://github.com/DSKSD/DeepNLP-models-Pytorch/blob/master/images/10.dmn-architecture.png" width='500'>

```
└── finch/tensorflow1/question_answering/babi
	│
	├── data
	│   └── make_data.ipynb           		# step 1. run this to generate vocabulary: word.txt 
	│   └── qa5_three-arg-relations_train.txt       # one complete example of babi dataset
	│   └── qa5_three-arg-relations_test.txt	# one complete example of babi dataset
	│
	├── vocab
	│   └── word.txt                  		# complete list of words in vocabulary
	│	
	└── main              
		└── dmn_train.ipynb
		└── dmn_serve.ipynb
		└── attn_gru_cell.py
```

* Task: [bAbI](https://research.fb.com/downloads/babi/)（English Data）

	* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/data/qa5_three-arg-relations_test.txt)
	
	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/data/make_data.ipynb)
	
	* Model: [Dynamic Memory Network](https://arxiv.org/abs/1603.01417)
	
		* TensorFlow 1
		
			* [\<Notebook> DMN -> 99.4% Testing Accuracy](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/main/dmn_train.ipynb)
			
			* [Inference](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/main/dmn_serve.ipynb)

---

## Text Processing Tools

* Word Matching

	* Chinese

		* [\<Notebook>: Regex Rule Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/python/regex/zhcn_extract.ipynb)

* Word Segmentation

	* Chinese
	
		* Jieba TensorFlow Op [\<Notebook>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/custom_op/tf_jieba.ipynb)

* Topic Modelling

	* Data: [2373 Lines of Book Titles](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/spark/topic_modelling/book_titles/all_book_titles.txt)（English Data）

		* Model: TF-IDF + LDA
		
			* PySpark
			
				* [\<Notebook> TF + IDF + LDA](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/spark/topic_modelling/book_titles/lda.ipynb)

			* Sklearn + pyLDAvis
			
				* [\<Notebook> TF + IDF + LDA](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/sklearn/topic_modelling/book_titles/lda.ipynb)
				
				* [\<Notebook> Visualization](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/sklearn/topic_modelling/book_titles/lda.html#topic=1&lambda=1&term=)

---

## Recommender System

<img src="https://github.com/PaddlePaddle/book/blob/develop/05.recommender_system/image/rec_regression_network.png" width='500'>

```
└── finch/tensorflow1/recommender/movielens
	│
	├── data
	│   └── make_data.ipynb           		# run this to generate vocabulary
	│
	├── vocab
	│   └── user_job.txt
	│   └── user_id.txt
	│   └── user_gender.txt
	│   └── user_age.txt
	│   └── movie_types.txt
	│   └── movie_title.txt
	│   └── movie_id.txt
	│	
	└── main              
		└── dnn_softmax.ipynb
		└── ......
```

* Task: [Movielens 1M](https://grouplens.org/datasets/movielens/1m/)（English Data）
	
        Training Data: 900228, Testing Data: 99981, Users: 6000, Movies: 4000, Rating: 1-5

	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/data/make_data.ipynb)

		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/data/train.txt)

	* Model: [Fusion](https://www.paddlepaddle.org.cn/documentation/docs/en/1.5/beginners_guide/basics/recommender_system/index_en.html)
	
		* TensorFlow 1
		
			 > MAE: Mean Absolute Error

			* [\<Notebook> Fusion + Sigmoid ->](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/main/dnn_sigmoid.ipynb)

 				0.663 Testing MAE

			* [\<Notebook> Fusion + Sigmoid + Cyclical LR ->](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/main/dnn_sigmoid_clr.ipynb)

 				0.661 Testing MAE

			* [\<Notebook> Fusion + Softmax ->](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/main/dnn_softmax.ipynb)

 				0.633 Testing MAE

			* [\<Notebook> Fusion + Softmax + Cyclical LR ->](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/recommender/movielens/main/dnn_softmax_clr.ipynb)

 				0.628 Testing MAE

				The MAE results seem better than the [all the results here](http://mymedialite.net/examples/datasets.html) and [all the results here](https://test.pypi.org/project/scikit-surprise/)

---

## Multi-turn Dialogue Rewriting

<img src="https://pic1.zhimg.com/80/v2-d80efd57b81c6ece955a247ca7247db4_1440w.jpg" width="600">

```
└── finch/tensorflow1/multi_turn_rewrite/chinese/
	│
	├── data
	│   └── make_data.ipynb         # run this to generate vocab, split train & test data, make pretrained embedding
	│   └── corpus.txt		# original data downloaded from external
	│   └── train_pos.txt		# processed positive training data after {make_data.ipynb}
	│   └── train_neg.txt		# processed negative training data after {make_data.ipynb}
	│   └── test_pos.txt		# processed positive testing data after {make_data.ipynb}
	│   └── test_neg.txt		# processed negative testing data after {make_data.ipynb}
	│
	├── vocab
	│   └── cc.zh.300.vec		# fastText pretrained embedding downloaded from external
	│   └── char.npy		# chinese characters and their embedding values (300 dim)	
	│   └── char.txt		# list of chinese characters used in this project 
	│	
	└── main              
		└── baseline_lstm_train.ipynb
		└── baseline_lstm_export.ipynb
		└── baseline_lstm_predict.ipynb
```

* Task: 20k 腾讯 AI 研发数据（Chinese Data）
	
        data split as: training data (positive): 18986, testing data (positive): 1008
	
	* [\<Text File>: Full Data](https://github.com/chin-gyou/dialogue-utterance-rewriter/blob/master/corpus.txt)
	
	* [\<Notebook>: Make Data & Vocabulary & Pretrained Embedding](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/data/make_data.ipynb)

			There are six incorrect data and we have deleted them

		* [\<Text File>: Positive Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/data/train_pos.txt)

	* Model: [RNN Seq2Seq + Attention](https://arxiv.org/abs/1409.0473) + [Dynamic Memory](https://arxiv.org/abs/1603.01417)

		* TensorFlow 1

			* Training with (positive + negative) data

					Training data = 2 * 18986 because of 1:1 Negative Sampling

				* [\<Text File>: Negative Data Example](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/data/train_neg.txt)
			
				* [\<Notebook> LSTM Seq2Seq + Attention + Multi-hop Memory + Beam Search](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_lstm_train_clr.ipynb)

					-> Exact Match: 56.2%, &nbsp; BLEU-1: 94.6, &nbsp; BLEU-2: 89.1, &nbsp; BELU-4: 78.5

				* [\<Notebook> GRU Seq2Seq + Attention + Multi-hop Memory + Beam Search](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_gru_train_clr.ipynb)

					-> Exact Match: 56.6%, &nbsp; BLEU-1: 94.5, &nbsp; BLEU-2: 88.9, &nbsp; BELU-4: 78.3

				* [\<Notebook> GRU Seq2Seq + Multi-Attention + Multi-hop Memory + Beam Search](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_gru_train_clr_multi_attn.ipynb)

					-> Exact Match: 56.2%, &nbsp; BLEU-1: 95.0, &nbsp; BLEU-2: 89.5, &nbsp; BELU-4: 78.9

			* Training with only positive data

				* [\<Notebook> GRU Seq2Seq + Attention + Multi-hop Memory + Beam Search](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_gru_train_clr_only_pos.ipynb)

					-> Exact Match: 57.6%, &nbsp; BLEU-1: 94.4, &nbsp; BLEU-2: 88.6, &nbsp; BELU-4: 77.5

				* [\<Notebook> GRU Seq2Seq + Multi-Attention + Multi-hop Memory + Beam Search](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_gru_train_clr_multi_attn_only_pos.ipynb)

					-> Exact Match: 58.2%, &nbsp; BLEU-1: 94.0, &nbsp; BLEU-2: 88.0, &nbsp; BELU-4: 76.5

	* Model: [RNN Pointer Networks](https://arxiv.org/abs/1506.03134)

		* TensorFlow 1

			Pointer Net returns probability distribution, therefore no need to do softmax again in beam search

			Go to beam search [source code](https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/seq2seq/python/ops/beam_search_decoder.py), replace this line

			```python
			step_log_probs = nn_ops.log_softmax(logits)
			```

			with this line

			```python
			step_log_probs = math_ops.log(logits)
			```

			* Training with only positive data

				* [\<Notebook> GRU Pointer Net](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/pointer_gru_train_clr_only_pos.ipynb)

					-> Exact Match: 59.2%, &nbsp; BLEU-1: 93.9, &nbsp; BLEU-2: 87.7, &nbsp; BELU-4: 76.1

			* Training with (positive + negative) data

				* [\<Notebook> GRU Pointer Net](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/pointer_gru_train_clr.ipynb)

					-> Exact Match: 59.2%, &nbsp; BLEU-1: 93.2, &nbsp; BLEU-2: 87.7, &nbsp; BELU-4: 77.2

				* [\<Notebook> GRU Pointer Net + Multi-Attention](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/pointer_gru_train_clr_multi_attn.ipynb)

					-> Exact Match: 58.7%, &nbsp; BLEU-1: 94.5, &nbsp; BLEU-2: 89.0, &nbsp; BELU-4: 78.8

					This result (only RNN, without BERT) is better than [the result here](https://github.com/liu-nlper/dialogue-utterance-rewriter) with BERT

	* Python Inference（基于 Python 的推理）

		* [\<Notebook> Export](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_lstm_export.ipynb)
		
		* [\<Notebook> Inference](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/tensorflow1/multi_turn_rewrite/chinese/main/baseline_lstm_predict.ipynb)
	
	* Java Inference（基于 Java 的推理）

		* [\<Notebook> Inference](https://github.com/zhedongzheng/tensorflow-nlp/blob/master/finch/java/MultiDialogInference/src/ModelInference.java)

```
└── MultiDialogInference
	│
	├── data
	│   └── baseline_lstm_export/
	│   └── char.txt
	│   └── libtensorflow-1.14.0.jar
	│   └── tensorflow_jni.dll
	│
	└── src              
	    └── ModelInference.java
```

## Knowledge Base Question Answering

<img src="https://upload-images.jianshu.io/upload_images/17747892-e994edc3518b2d58.png?imageMogr2/auto-orient/strip|imageView2/2/w/880" height="350">

* Rule-based System（基于规则的系统）
	
	For example, we want to answer the following questions:
	
	```
		宝马是什么?  /  what is BMW?
        	我想了解一下宝马  /  i want to know about the BMW
        	给我介绍一下宝马  /  please introduce the BMW to me
		宝马这个牌子的汽车怎么样?  /  how is the car of BMW group?
        	宝马如何呢?  /  how is the BMW?
        	宝马汽车好用吗?  /  is BMW a good car to use?
        	宝马和奔驰比怎么样?  /  how is the BMW compared to the Benz?
        	宝马和奔驰比哪个好?  /  which one is better, the BMW or the Benz?
        	宝马和奔驰比哪个更好?  /  which one is even better, the BMW or the Benz?
	```
	
	* [refo](https://github.com/machinalis/refo) + [jieba](https://github.com/fxsjy/jieba): &nbsp; &nbsp; [Example](https://nbviewer.jupyter.org/github/zhedongzheng/tensorflow-nlp/blob/master/finch/python/kbqa/rule_based_qa.ipynb)
