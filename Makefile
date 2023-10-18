SHELL := /bin/bash

.PONY: main_experiment

###############################################################################################################
###############################################################################################################

main_experiment: bert roberta distilbert distilroberta bertweet xlnet albert

bert: bert-base-uncased bert-large-uncased 

distilbert: distilbert-base-uncased

distilroberta: distilroberta-base

roberta: roberta-base roberta-large

bertweet: bertweet-base bertweet-large

xlnet: xlnet-base-cased xlnet-large-cased

zari: zari-bert-cda zari-bert-dropout

albert: albert-base-v2 albert-large-v2

###############################################################################################################
###############################################################################################################

bert-base-uncased:
	python evaluate.py $(args) --model_name="bert-base-uncased" 

bert-large-uncased:
	python evaluate.py $(args) --model_name="bert-large-uncased"

distilbert-base-uncased:
	python evaluate.py $(args) --model_name="distilbert-base-uncased" 

distilroberta-base:
	python evaluate.py $(args) --model_name="distilroberta-base" 

roberta-base:
	python evaluate.py $(args) --model_name="roberta-base" 

roberta-large:
	python evaluate.py $(args) --model_name="roberta-large" 

bertweet-base:
	python evaluate.py $(args) --model_name="vinai/bertweet-base" 

bertweet-large:
	python evaluate.py $(args) --model_name="vinai/bertweet-large" 

xlnet-base-cased:
	python evaluate.py $(args) --model_name="xlnet-base-cased" 

xlnet-large-cased:
	python evaluate.py $(args) --model_name="xlnet-large-cased" 

albert-base-v2:
	python evaluate.py $(args) --model_name="albert-base-v2" 

albert-large-v2:
	python evaluate.py $(args) --model_name="albert-large-v2"
