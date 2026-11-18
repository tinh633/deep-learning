"""Tokenize one sentence, run an SST-2 classifier, and save the evidence."""

import argparse
import json

from config import DEFAULT_SENTENCE, MAX_LENGTH, MODEL_ID, RESULT_FILE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE)
    args = parser.parse_args()
    if not args.sentence.strip():
        parser.error("--sentence must contain text")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    inputs = tokenizer(
        args.sentence, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    token_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probabilities = logits.softmax(dim=-1).tolist()

    scores = {
        model.config.id2label[index]: score for index, score in enumerate(probabilities)
    }
    prediction = max(scores, key=scores.__getitem__)
    result = {
        "model_id": MODEL_ID,
        "sentence": args.sentence,
        "tokens": tokens,
        "token_ids": token_ids,
        "attention_mask": inputs["attention_mask"][0].tolist(),
        "max_length": MAX_LENGTH,
        "logits": {
            model.config.id2label[index]: value for index, value in enumerate(logits.tolist())
        },
        "scores": scores,
        "predicted_label": prediction,
        "confidence": scores[prediction],
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved {RESULT_FILE}")


if __name__ == "__main__":
    main()
