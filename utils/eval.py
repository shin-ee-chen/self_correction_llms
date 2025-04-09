from math_verify import verify
import re

def per_sample_verification(preds, ground_truth):
        scores = [verify(str(ground_truth), str(pred)) for pred in preds]
        return scores
