"""Exercise 1: model selection, tokenization, inference and saved evidence."""
import argparse
from datetime import datetime, timezone
from transformers import AutoTokenizer, pipeline
from config import OUTPUTS, SENTIMENT_MODEL
from utils import write_json

DEFAULT_SENTENCES = [
    'The film started slowly, but I absolutely loved its ending.',
    'This was a complete waste of two hours; the plot made no sense.',
    'The acting was fine, I guess, though nothing special.',
    'I did not hate it.',
]

def main():
    parser = argparse.ArgumentParser(description='Pretrained binary sentiment inference')
    parser.add_argument('--sentence', action='append',
                        help='Sentence to analyse; repeat the flag for several sentences')
    args = parser.parse_args()
    sentences = [s for s in (args.sentence or DEFAULT_SENTENCES) if s.strip()]
    if not sentences:
        raise SystemExit('Provide at least one non-empty sentence.')
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    classifier = pipeline('sentiment-analysis', model=SENTIMENT_MODEL, tokenizer=tokenizer,
                          device=-1, truncation=True, max_length=512)
    results = []
    for sentence in sentences:
        encoded = tokenizer(sentence)
        prediction = classifier(sentence)[0]
        results.append({
            'sentence': sentence,
            'tokens': tokenizer.tokenize(sentence), 'token_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'predicted_label': prediction['label'], 'model_score': float(prediction['score'])})
    # Top-level fields mirror the first sentence so earlier consumers keep working.
    evidence = {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'model_id': SENTIMENT_MODEL, **results[0], 'results': results,
        'note': 'Model score is not a calibrated guarantee of correctness.'}
    output = OUTPUTS / 'exercise1' / 'sentiment_result.json'
    write_json(output, evidence)
    for r in results:
        print('Sentence:', r['sentence'])
        print('  Tokens:', r['tokens'])
        print('  Token IDs:', r['token_ids'])
        print('  Prediction:', r['predicted_label'], '| score:', round(r['model_score'], 4))
    print('Saved:', output)

if __name__ == '__main__':
    main()
