from math_verify import verify

def per_sample_verification(preds, ground_truth):

        scores = [verify(ground_truth, pred) for pred in preds]
        return scores
