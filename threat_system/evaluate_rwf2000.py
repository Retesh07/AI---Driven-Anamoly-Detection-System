#!/usr/bin/env python3
"""Evaluate the violence model on extracted RWF-2000 validation features.

Expected layout: FEATURES/{Fight,NonFight}/*.npy, each shaped (60, 126).
This intentionally evaluates only RWF-2000's fight/non-fight task; dancing and
sports are not output classes and belong in a separate hard-negative set.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--mean', required=True, type=Path)
    parser.add_argument('--std', required=True, type=Path)
    parser.add_argument('--threshold', type=float, default=0.65)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--min-f1', type=float)
    parser.add_argument('--max-fpr', type=float)
    return parser.parse_args()


def metrics(labels, predictions):
    tp = sum(y == p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        'samples': len(labels), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': (tp + tn) / len(labels),
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'false_positive_rate': fp / (fp + tn) if fp + tn else 0.0,
    }


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit('--threshold must be between 0 and 1')

    import numpy as np
    import torch
    from violence.model import ViolenceDetectorV3

    for path in (args.features, args.model, args.mean, args.std):
        if not path.exists():
            raise SystemExit(f'Not found: {path}')

    mean = np.load(args.mean).astype(np.float32)
    std = np.maximum(np.load(args.std).astype(np.float32), 1e-6)
    model = ViolenceDetectorV3().to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.eval()

    labels, predictions, skipped = [], [], 0
    with torch.no_grad():
        for class_name, label in (('Fight', 1), ('NonFight', 0)):
            for path in sorted((args.features / class_name).glob('*.npy')):
                sample = np.load(path).astype(np.float32)
                if sample.shape != (60, 126):
                    skipped += 1
                    continue
                logit, _ = model(torch.from_numpy((sample - mean) / std).unsqueeze(0).to(args.device))
                probability = float(torch.sigmoid(logit).item())
                labels.append(label)
                predictions.append(int(probability >= args.threshold))

    if not labels:
        raise SystemExit(f'No valid RWF-2000 features found under {args.features}')
    report = metrics(labels, predictions)
    report.update({'threshold': args.threshold, 'skipped': skipped})
    print(json.dumps(report, indent=2))

    failed = ((args.min_f1 is not None and report['f1'] < args.min_f1) or
              (args.max_fpr is not None and report['false_positive_rate'] > args.max_fpr))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
