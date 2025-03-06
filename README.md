# HalluCounter
# Installation
```
pip install hallucounter
```

# Hallucounter Usage

```
### Import

from hallucounter.detector import load_classifier, get_predictions, get_pipeline_output
from hallucounter.nli_scores import load_nli_model, get_nli_scores

### Load Classifier
clf = load_classifier('path_to_classification_model_file')

### Load NLI Cross-Encoder
nli_model, nli_tokenizer = load_nli_model() # takes optional parameter model_name. Unless specified, it uses nli-deberta-v3-large by default

### Define Query and Responses
query = "Which British athlete won the 100 m. at the 1924 Olympics?"
responses = ["Harold Abrahams.", "Harold Abrahams.","Harold Abrahams.","Harold Abrahams.","Harold Abrahams."] # Here, K=5

nli_scores = get_nli_scores(query, responses, nli_model, nli_tokenizer)

### Get the Hallucounter pipeline output
overall_prediction, confidence_score, optimal_response = get_pipeline_output(nli_scores, clf, responses)
print("Overall Prediction:", overall_prediction)
print("Optimal Response:", optimal_response)
print("Confidence Score:", confidence_score)
```
