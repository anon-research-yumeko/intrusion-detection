"""Command-line entry point for the chipoccon evaluation workflow."""

import argparse
import os
from typing import Optional, Sequence



import pathlib
import sys

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT.parent))
from chipoccon.datasets import SUPPORTED_DATASETS, resolve_data_dir
from chipoccon.evaluate import evaluate_grid

__all__ = ["build_parser", "parse_args", "main"]


def build_parser() -> argparse.ArgumentParser:
	"""Construct the argument parser shared by CLI and programmatic entry points."""

	parser = argparse.ArgumentParser(
		description="Comprehensive evaluation for OCCon plus optional Chip fusion",
	)

	parser.add_argument(
		"--dataset",
		type=str,
		default="unsw",
		choices=SUPPORTED_DATASETS,
		help="Dataset to preprocess: unsw, cicids2017, or cicids2018",
	)
	parser.add_argument(
		"--data_dir",
		type=str,
		default="auto",
		help="Path to dataset files. Use 'auto' to pick the default directory for the chosen dataset.",
	)
	parser.add_argument(
		"--out_dir",
		type=str,
		default="chipoccon_outputs",
		help="Where to save CSV outputs",
	)
	parser.add_argument(
		"--array_id",
		type=int,
		default=0,
		help="Index of this shard (0-based). Used only if --array_shards > 1",
	)
	parser.add_argument(
		"--array_shards",
		type=int,
		default=1,
		help="Number of shards to split the OCCon train grid into",
	)

	parser.add_argument(
		"--seeds",
		type=int,
		nargs="+",
		default=[13],
		help="List of seeds to evaluate",
	)

	parser.add_argument(
		"--cut_off_list",
		type=float,
		nargs="+",
		default=[1.0],
		help="Cutoff over [0, 1] of max timestamp",
	)

	parser.add_argument(
		"--embedding_dim_list",
		type=int,
		nargs="+",
		default=[128],
		help="Embedding dimensions for the OCCon encoder",
	)
	parser.add_argument(
		"--temperature_list",
		type=float,
		nargs="+",
		default=[0.1],
		help="OCCon temperatures",
	)
	parser.add_argument(
		"--batch_size_list",
		type=int,
		nargs="+",
		default=[256],
		help="Batch sizes for OCCon encoder training (grid)",
	)

	parser.add_argument(
		"--k_skmeans_list",
		type=int,
		nargs="+",
		default=[2, 3, 5, 10, 20, 30],
		help="Cluster counts for spherical KMeans",
	)

	parser.add_argument(
		"--f1_percentile",
		type=str,
		default="auto",
		help="'auto' uses the benign ratio; otherwise supply a numeric percentile (e.g. 75.5)",
	)

	parser.add_argument("--batch_size", type=int, default=256, help="Training batch size for OCCon")
	parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for OCCon")
	parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for OCCon")
	parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs for OCCon")
	parser.add_argument(
		"--min_epochs",
		type=int,
		default=30,
		help="Train at least this many epochs before early stopping",
	)
	parser.add_argument(
		"--early_stopping_patience",
		type=int,
		default=10,
		help="Early stopping patience after min_epochs",
	)
	parser.add_argument(
		"--num_workers",
		type=int,
		default=0,
		help="DataLoader worker count",
	)

	parser.add_argument(
		"--enable_chip",
		action="store_true",
		dest="enable_chip",
		help="Enable OCCon+Chip combined evaluation",
	)

	return parser


def parse_args(argv: Optional[Sequence[str]] = None):
	"""Parse CLI arguments; accepts an override argv for programmatic use."""

	parser = build_parser()
	return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
	"""Entry point used by the console script stub and python -m chipoccon.main."""

	args = parse_args(argv)
	args.data_dir = resolve_data_dir(args.dataset, args.data_dir)

	if args.array_shards > 1:
		args.out_dir = os.path.join(args.out_dir, f"shard_{args.array_id}_of_{args.array_shards}")

	os.makedirs(args.out_dir, exist_ok=True)
	evaluate_grid(args)


if __name__ == "__main__":
	main()
