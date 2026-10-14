"""2026 extension: explicit identities, fixed split and strict evaluation."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

CATEGORIES = ['Cargo plane', 'Small car', 'Bus', 'Truck', 'Motorboat',
              'Fishing vessel', 'Dump truck', 'Excavator', 'Building',
              'Helipad', 'Storage tank', 'Shipping container', 'Pylon']

def records_from_annotations(document):
    images = list(document['images'].values())
    annotations = list(document['annotations'].values())
    by_id = {}
    for ann in annotations:
        key = ann['image_id']
        if key in by_id:
            raise ValueError('Classification expects one annotation per image')
        by_id[key] = ann
    records, ids, filenames = [], set(), set()
    for image in images:
        key, filename = image['image_id'], image['filename']
        if key in ids or filename in filenames:
            raise ValueError('Duplicate image identity')
        label = by_id[key]['category_id']
        if label not in CATEGORIES:
            raise ValueError('Unknown class')
        ids.add(key); filenames.add(filename)
        records.append(dict(image_id=key, filename=filename, label=label))
    if set(by_id) != ids:
        raise ValueError('Unmatched annotation')
    return records

def historical_split(records, seed=42):
    # sklearn ShuffleSplit: ceil(n * .1) validation rows, permutation from MT19937.
    if len(records) < 2:
        raise ValueError('At least two images required')
    order = np.random.RandomState(seed).permutation(len(records))
    n_valid = math.ceil(len(records) * .1)
    return ([records[i] for i in order[n_valid:]],
            [records[i] for i in order[:n_valid]])

def accumulation_group_size(index, batches, accumulation):
    if accumulation < 1 or not 0 <= index < batches:
        raise ValueError('Invalid accumulation bounds')
    return min(accumulation, batches - (index // accumulation) * accumulation)

def accumulation_boundary(index, batches, accumulation):
    accumulation_group_size(index, batches, accumulation)
    return (index + 1) % accumulation == 0 or index + 1 == batches

def evaluate_records(expected, predictions):
    def keyed(rows):
        result = {}
        for row in rows:
            key = row['image_id']
            if key in result:
                raise ValueError('Duplicate prediction or label ID')
            result[key] = row
        return result
    truth, pred = keyed(expected), keyed(predictions)
    if not truth or set(truth) != set(pred):
        raise ValueError('Prediction IDs must exactly match the evaluated split')
    confusion = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=int)
    errors = []
    for key, row in truth.items():
        other = pred[key]
        if row['filename'] != other['filename']:
            raise ValueError('Filename does not match image ID')
        if row['label'] not in CATEGORIES or other['predicted_label'] not in CATEGORIES:
            raise ValueError('Unknown class')
        i, j = CATEGORIES.index(row['label']), CATEGORIES.index(other['predicted_label'])
        confusion[i, j] += 1
        if i != j:
            errors.append(dict(image_id=key, filename=row['filename'],
                               label=row['label'], predicted_label=other['predicted_label']))
    per_class = {}
    for i, label in enumerate(CATEGORIES):
        tp = int(confusion[i, i]); support = int(confusion[i].sum())
        predicted = int(confusion[:, i].sum())
        precision = tp / predicted if predicted else 0.
        recall = tp / support if support else 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return dict(n=len(truth), accuracy=float(np.trace(confusion) / len(truth)),
                macro_f1=float(np.mean([v['f1'] for v in per_class.values()])),
                macro_f1_policy='Fixed 13 classes; zero for undefined F1',
                classes=CATEGORIES, confusion_matrix=confusion.tolist(),
                per_class=per_class, errors=errors)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('labels'); parser.add_argument('predictions')
    args = parser.parse_args()
    with open(args.labels) as f: labels = list(csv.DictReader(f))
    predictions = json.loads(Path(args.predictions).read_text())
    print(json.dumps(evaluate_records(labels, predictions), indent=2))
