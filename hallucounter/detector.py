import os
# os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import numpy as np

import json
from joblib import load
# import torch
import pandas as pandas
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sklearn
import argparse

def load_classifier(classifier_path):
  clf = load(classifier_path)
  return clf

def get_predictions(nli_features,clf):
    predictions=clf.predict(nli_features)
    return predictions
# predictions=get_predictions(nli_features,clf)

"""# Confidence Score and Optimal Response"""

def get_pipeline_output(nli_scores,clf,responses):
    predicted_labels=[]
    overall_predictions=[]
    confidence_scores=[]
    optimal_responses=[]

    predictions=get_predictions(nli_scores,clf)
    predicted_labels.append(predictions)

    count_1 = list(predictions).count('1')
    count_0 = list(predictions).count('0')
    overall_prediction='1' if count_1> count_0 else '0'
    overall_predictions.append(overall_prediction)

    # ##### made changes ####################
    def get_ent_cont_scores(feature_name,score):
        if feature_name=='NLI_Features':
            contradiction_score=0.3*score[2]+0.7*score[5]
            entailment_score=0.3*score[0]+0.7*score[3]
        elif feature_name=='NLI_Features_V1':
            contradiction_score=score[2]
            entailment_score=score[0]
        elif feature_name=='NLI_Features_V2':
            contradiction_score=score[2]
            entailment_score=score[0]
        elif feature_name=='NLI_Features_V3':
            contradiction_score=0.3*score[1]+0.7*score[3]
            entailment_score=0.3*score[0]+0.7*score[2]
        else:
            contradiction_score=0.3*score[0]+0.7*score[1]
            entailment_score=0
        return entailment_score, contradiction_score

    if overall_prediction=='1':
        hallu_responses=[r for idx,r in enumerate(responses) if predictions[idx]=='1']
        hallu_nli_scores=[s for idx,s in enumerate(nli_scores) if predictions[idx]=='1']
        confidence_score=count_1/len(predictions)
        contradiction_scores=[]
        for score in hallu_nli_scores:
            _, cont_score =get_ent_cont_scores(feature_name,score)
            contradiction_scores.append(cont_score)
        optimal_response_index=np.argmin(contradiction_scores)
        optimal_response=hallu_responses[optimal_response_index]
    else:
        non_hallu_responses=[r for idx,r in enumerate(responses) if predictions[idx]=='0']
        non_hallu_nli_scores=[s for idx,s in enumerate(nli_scores) if predictions[idx]=='0']
        confidence_score=count_0/len(predictions)
        entailment_scores=[]
        for score in non_hallu_nli_scores:
            ent_score, _ =get_ent_cont_scores(feature_name,score)
            entailment_scores.append(ent_score)
        optimal_response_index=np.argmax(entailment_scores)
        optimal_response=non_hallu_responses[optimal_response_index]
    
    
    confidence_scores.append(confidence_score)
    # optimal_response=responses[optimal_response_index]
    optimal_responses.append(optimal_response)


    return overall_predictions[0], confidence_scores[0], optimal_responses[0]


def main():
    parser = argparse.ArgumentParser(description="Detect hallucinations in LLM-generated responses.")
    parser.add_argument("query", type=str, help="The query to be processed")
    parser.add_argument("responses", nargs='+', help="List of responses to the query")
    parser.add_argument("classifier_path", type=str, help="Path to classifier file")
    args = parser.parse_args()

    clf = load_classifier(args.classifier_path)
    # overall_prediction, confidence_score, optimal_response = get_pipeline_output(get_nli_scores(args.query, args.responses), clf, args.responses)

    # print("Overall Prediction:", overall_prediction)
    # print("Optimal Response:", optimal_response)
    # print("Confidence Score:", confidence_score)

if __name__ == "__main__":
    main()
