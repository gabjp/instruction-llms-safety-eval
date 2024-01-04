from evals import AbsoluteHarmfulnessPredictor, ConversationBuilder
from evals import ContentModeration
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--outs_dir", type=str, default="")
parser.add_argument("--open_ai_key", type=str, default="")

DS_NAMES =[ "I-Alpaca.json",  "I-CoNa.json",  "I-Controversial.json",  "I-MaliciousInstructions.json",  "I-PhysicalSafetySafe.json",  "I-PhysicalSafetyUnsafe.json" ]

def main():
    args = parser.parse_args()
    os.environ["OPEN_AI_KEY"] = args.open_ai_key

    for ds_name in DS_NAMES:
        print(f"Evaluating {ds_name}")
        path = os.path.join(args.outs_dir, ds_name)
        f = open(path) 
        data = json.load(f)
        f.close()

        user_texts = data["instructions"]
        assistant_texts = data["outputs"]

        print("starting model evaluation")
        setup = "redteam"  # or "redteam-osst"
        harmfulness_predictor = AbsoluteHarmfulnessPredictor(setup, device="cuda:0")
        harmfulness_scores = harmfulness_predictor.predict(user_texts, assistant_texts)
        print(sum(harmfulness_scores) / len(harmfulness_scores))

        print("starting OpenAI evaluation")
        cm = ContentModeration()
        scores = cm.content_moderation(assistant_texts)
        print(sum(scores)/len(scores))
        
    return 

if __name__ == "__main__":
    main()