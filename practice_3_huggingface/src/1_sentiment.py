"""Exercise 1: model selection, tokenization, inference and saved evidence."""
import argparse
import json
from datetime import datetime, timezone
from transformers import AutoTokenizer, pipeline
from config import OUTPUTS, SENTIMENT_MODEL
from utils import write_json

def main():
    parser = argparse.ArgumentParser(description='Pretrained binary sentiment inference')
    parser.add_argument('--sentence', default='The film started slowly, but I absolutely loved its ending.')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    encoded = tokenizer(args.sentence)
    classifier = pipeline('sentiment-analysis', model=SENTIMENT_MODEL, tokenizer=tokenizer, device=-1)
    result = classifier(args.sentence)[0]
    evidence = {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'model_id': SENTIMENT_MODEL, 'sentence': args.sentence,
        'tokens': tokenizer.tokenize(args.sentence), 'token_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'predicted_label': result['label'], 'model_score': float(result['score']),
        'note': 'Model score is not a calibrated guarantee of correctness.'}
    output = OUTPUTS / 'exercise1' / 'sentiment_result.json'
    write_json(output, evidence)
    print('Sentence:', args.sentence)
    print('Tokens:', evidence['tokens'])
    print('Token IDs:', evidence['token_ids'])
    print('Prediction:', result['label'])
    print('Confidence:', round(result['score'], 4))
    print('Saved:', output)

if __name__ == '__main__':
    main()
