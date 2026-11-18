"""Optional inference demonstration using a saved fine-tuned model; no training."""
import argparse
from transformers import pipeline
from config import RUNS, EVAL_RUN

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sentence',default='The acting was memorable and the film was excellent.')
    args=parser.parse_args()
    model=RUNS/EVAL_RUN/'model'
    if not model.exists():raise SystemExit('No saved model: '+str(model))
    classifier=pipeline('text-classification',model=str(model),tokenizer=str(model),device=-1)
    prediction=classifier(args.sentence)[0]
    print('Run:',EVAL_RUN,'Input:',args.sentence,'Prediction:',prediction)

if __name__=='__main__':main()
