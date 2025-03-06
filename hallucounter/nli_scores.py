import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
"""# NLI Scores Extraction"""
def load_nli_model(model_name='cross-encoder/nli-deberta-v3-large'):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Load cross-encoder model and tokenizer
  nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
  nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
  return nli_model, nli_tokenizer

def get_nli_scores(query,responses,nli_model, nli_tokenizer):
    # Function to calculate cross-encoder similarity scores
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def cross_encoder_scores(hypothesis, premise):
        features = nli_tokenizer([hypothesis], [premise], padding=True, truncation=True, return_tensors="pt").to(device)
        nli_model.eval()
        with torch.no_grad():
            scores = nli_model(**features).logits
        return scores.cpu().numpy()[0].tolist()  # Move tensor to CPU and convert to numpy

    question = query
    llm_responses = responses
    each_label_scores = []
    for i in range(len(llm_responses)):
        query_response_ce_score = cross_encoder_scores(question, llm_responses[i])
        final_CE_score = []
        entailment, neutral, contradiction = [], [], []
        for j in range(len(llm_responses)):
            if i != j:
                score = cross_encoder_scores(llm_responses[i], llm_responses[j])
                entailment.append(score[0])
                neutral.append(score[1])
                contradiction.append(score[2])
        final_ent_score = sum(entailment)/len(entailment)
        final_CE_score.append(final_ent_score)
        final_neu_score = sum(neutral)/len(neutral)
        final_CE_score.append(final_neu_score)
        final_cnt_score = sum(contradiction)/len(contradiction)
        final_CE_score.append(final_cnt_score)
        one_label_score = query_response_ce_score + final_CE_score
        each_label_scores.append(one_label_score)

    nli_scores=[] # taking only ec-ec features for now (change it according to the classifier that we have used)

    for feat in each_label_scores:
      nli_scores.append([value for idx,value in enumerate(feat) if idx not in [1,4]])

    return nli_scores # returns a nested list of NLI scores (one list each for k responses)


def main():
    parser = argparse.ArgumentParser(description="Detect hallucinations in LLM-generated responses.")
    parser.add_argument("query", type=str, help="The query to be processed")
    parser.add_argument("responses", nargs='+', help="List of responses to the query")
    parser.add_argument("classifier_path", type=str, help="Path to classifier file")
    args = parser.parse_args()

    # clf = load_classifier(args.classifier_path)
    # overall_prediction, confidence_score, optimal_response = get_pipeline_output(get_nli_scores(args.query, args.responses), clf, args.responses)

    # print("Overall Prediction:", overall_prediction)
    # print("Optimal Response:", optimal_response)
    # print("Confidence Score:", confidence_score)

if __name__ == "__main__":
    main()
