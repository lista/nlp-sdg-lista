"""
This is a boilerplate pipeline 'text_comprehension'
generated using Kedro 0.18.2
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pyspark.sql import DataFrame
from allennlp.predictors.predictor import Predictor

model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
tokenizer = AutoTokenizer.from_pretrained('t5-base')
qna_predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2020.03.19.tar.gz')
ner_predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz')


def summarize_text(data: pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes in a dataframe.
    Then it returns a dataframe that creates summarized column of the text column.
    Args:
        source training data
    
    Returns:
        data with a "text" containing the summarized text
    '''
    data = data.head(3)
    result = []
    for i in range(len(data["text"])):
        result.append(tokenizer.encode("summarize: "+data["text"][i], return_tensors='pt', 
                                      max_length=tokenizer.model_max_length,
                                      truncation=True))
    
    data["tokens_input"] = result
    
    
    result_1 = []
    for i in range(len(data["tokens_input"])):
        result_1.append(model.generate(data["tokens_input"][i], min_length=80, 
                                      max_length=150, length_penalty=15, 
                                     early_stopping=True))
    
    data["summary_ids"] = result_1
    
    
    result_2 = []
    for i in range(len(data["summary_ids"])):
        result_2.append(tokenizer.decode((data["summary_ids"][i])[0], skip_special_tokens=True))
        
    
    data["summary"]=result_2
    
    data = data.drop(["tokens_input", "summary_ids"], axis=1)

    return data

def question_and_answer(data : pd.DataFrame,questions = "which resorces are mentioned?") -> pd.DataFrame:
        """question and answer node meant to produce answers to articles in a dataframe
        Args:
            data: Data containing a text column.
        Returns:
            data: a dataframe answering the asked question based on the articles in each row 
        """
        data = data.head(2)
        def qestions(data,question):
            result = []
            for i in range(len(data["text"])):
                result.append(qna_predictor.predict(passage=data["text"][i], question=question)["best_span_str"])
            return result

        data[questions] = qestions(data, question = questions)
        return data[questions]




def named_entity_recognition(data: pd.DataFrame) -> pd.DataFrame:
    data= data.head(5)
    def entity_recognition (sentence):
        location = []
        results =  ner_predictor.predict(sentence=sentence)
        for word, tag in zip(results["words"], results["tags"]):
            if tag != 'U-LOC':
                continue
            else:
                location.append(word)
        return location

    def entity_recognition_pe(sentence):
        organisation = []
        for nlp_model in nlp_models:
            results =  ner_predictor.predict(sentence=sentence)
            for word, tag in zip(results["words"], results["tags"]):
                if tag != 'U-ORG':
                    continue
                else:
                    # print([word])#(f"{word}")
                    organisation.append(word)
            return organisation
    result = []
    for i in range(len(data["text"])):
        result.append(list(set(entity_recognition(data["text"][i]))))
    re1 = []
    for i in range(len(data["text"])):
        re1.append(list(set(entity_recognition_pe(data["text"][i]))))
    data["location"]=result
    data["organisation"]=re1
    return data[["text","location","organisation"]]