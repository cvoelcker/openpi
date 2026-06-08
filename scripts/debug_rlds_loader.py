#!/usr/bin/env python3
"""Quick one-off RLDS/DROID data loader test.

Usage examples:
  python scripts/debug_rlds_loader.py --data-dir /datastor1/droid/ \
      --dataset droid_100 --version 1.0.0 --no-filter

This will construct the TF pipeline with a small shuffle buffer and attempt to
fetch a single batch while printing timestamps before/after each stage.
"""
import argparse
import logging
import time
from pprint import pformat

from openpi.training.droid_rlds_dataset import DroidRldsDataset, RLDSDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", default="droid_100")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shuffle-buffer-size", type=int, default=1000)
    parser.add_argument("--num-parallel", type=int, default=4)
    parser.add_argument("--no-filter", action="store_true", help="disable idle filter (faster)")
    parser.add_argument("--filter-dict", type=str, default=None, help="path or gs:// URL to filter dict JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    filter_path = None if args.no_filter else args.filter_dict
    dataset_cfg = RLDSDataset(name=args.dataset, version=args.version, weight=1.0, filter_dict_path=filter_path)

    logging.info("Constructing DroidRldsDataset (this may import TF)...")
    t0 = time.time()
    ds = DroidRldsDataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        datasets=[dataset_cfg],
        shuffle=False,
        shuffle_buffer_size=args.shuffle_buffer_size,
        num_parallel_reads=args.num_parallel,
        num_parallel_calls=args.num_parallel,
        her_gamma=None,
    )
    t1 = time.time()
    logging.info("Constructed dataset in %.3f s", t1 - t0)

    it = iter(ds)
    logging.info("Fetching first batch now...")
    t2 = time.time()
    try:
        batch = next(it)
    except Exception as e:
        logging.exception("Error while fetching first batch: %s", e)
        raise
    t3 = time.time()
    logging.info("Fetched batch in %.3f s", t3 - t2)

    # Print a compact summary of the batch
    try:
        summary = {k: (type(v), getattr(v, 'shape', None)) for k, v in batch.items()}
    except Exception:
        summary = str(batch)
    logging.info("Batch summary:\n%s", pformat(summary))


if __name__ == "__main__":
    main()
