"""Select an experiment by validation results, without opening test reports."""
from pathlib import Path
import argparse
import json
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common import read_json, write_json


def select(runs):
    records = []
    for run in runs:
        run = Path(run)
        metrics = read_json(run/'metrics.json')
        if metrics.get('test') is not None:
            raise ValueError('Test already exposed for ' + run.name + '; use a fresh held-out benchmark for model selection')
        records.append(dict(run=run.name, dataset_sha256=metrics['dataset_sha256'],
                            validation=metrics['validation'], best_epoch=metrics['best_epoch']))
    if not records or len({r['dataset_sha256'] for r in records}) != 1:
        raise ValueError('Experiments must share the exact dataset and split')
    chosen = max(records, key=lambda r:(r['validation']['top5'], r['validation']['mrr']))
    return dict(schema=1, rule='validation top5 then MRR; supplied order breaks ties', selected=chosen, candidates=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = select(a.runs)
    write_json(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
