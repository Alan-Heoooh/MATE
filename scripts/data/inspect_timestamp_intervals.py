#!/usr/bin/env python3
import argparse
from collections import Counter
from typing import Optional, Tuple

import numpy as np


def _find_dataset_path_by_name(h5file, name: str) -> Optional[str]:
    found = []

    def visitor(path, obj):
        if getattr(obj, "shape", None) is not None and path.split("/")[-1] == name:
            found.append(path)

    h5file.visititems(visitor)
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    # Prefer the shortest path (often the most canonical).
    found.sort(key=lambda p: (p.count("/"), len(p)))
    return found[0]


def _load_timestamp(h5_path: str, dataset_path: Optional[str]) -> Tuple[np.ndarray, str]:
    import h5py

    with h5py.File(h5_path, "r") as f:
        if dataset_path is None:
            dataset_path = _find_dataset_path_by_name(f, "timestamp")
            if dataset_path is None:
                raise KeyError(
                    "Could not find dataset named 'timestamp' in the HDF5 file. "
                    "Pass --dataset-path explicitly."
                )
        ts = np.asarray(f[dataset_path])
        return ts, dataset_path


def _summarize_intervals(intervals: np.ndarray) -> str:
    if intervals.size == 0:
        return "Not enough timestamps to compute intervals (need at least 2)."

    finite = intervals[np.isfinite(intervals)]
    if finite.size == 0:
        return "All computed intervals are non-finite."

    qs = np.quantile(finite, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    lines = [
        f"count: {finite.size}",
        f"min:   {qs[0]:.6g}",
        f"p1:    {qs[1]:.6g}",
        f"p5:    {qs[2]:.6g}",
        f"p50:   {qs[3]:.6g}",
        f"p95:   {qs[4]:.6g}",
        f"p99:   {qs[5]:.6g}",
        f"max:   {qs[6]:.6g}",
        f"mean:  {finite.mean():.6g}",
        f"std:   {finite.std():.6g}",
    ]

    neg = np.sum(finite < 0)
    zero = np.sum(finite == 0)
    if neg or zero:
        lines.append(f"negative intervals: {int(neg)}")
        lines.append(f"zero intervals:     {int(zero)}")

    return "\n".join(lines)


def _plot_interval_histogram(
    intervals: np.ndarray,
    title: str,
    out_path: Optional[str],
    show: bool,
    max_unique: int,
    logy: bool,
) -> None:
    finite = intervals[np.isfinite(intervals)]
    if finite.size == 0:
        raise ValueError("No finite intervals to plot.")

    # Most MATE timestamps are integer-like; round safely for counting.
    ints = np.rint(finite).astype(np.int64)
    counts = Counter(ints.tolist())
    if not counts:
        raise ValueError("No intervals to plot after rounding.")

    xs = np.array(sorted(counts.keys()), dtype=np.int64)
    ys = np.array([counts[int(x)] for x in xs], dtype=np.int64)

    if xs.size > max_unique:
        # Fall back to binned histogram when too many unique values.
        import matplotlib

        if out_path and not show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        bins = min(200, max(10, int(np.sqrt(finite.size))))
        plt.hist(finite, bins=bins, edgecolor="none")
        plt.title(title + f" (binned, {bins} bins)")
        plt.xlabel("interval")
        plt.ylabel("count")
        if logy:
            plt.yscale("log")
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=200)
        if show:
            plt.show()
        plt.close()
        return

    import matplotlib

    if out_path and not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.bar(xs, ys, width=0.9)
    plt.title(title)
    plt.xlabel("interval (rounded to int)")
    plt.ylabel("count")
    if logy:
        plt.yscale("log")
    plt.xticks(xs if xs.size <= 40 else xs[:: max(1, xs.size // 40)], rotation=0)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read lowdim.h5 and compute adjacent timestamp intervals (diff). "
            "By default, auto-detects dataset named 'timestamp'."
        )
    )
    parser.add_argument(
        "--h5",
        required=True,
        help="Path to lowdim.h5",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="HDF5 dataset path for timestamps, e.g. /timestamp (optional)",
    )
    parser.add_argument(
        "--print-first-n",
        type=int,
        default=0,
        help="Print first N intervals (optional)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot histogram of interval counts (bar chart for integer-like intervals).",
    )
    parser.add_argument(
        "--out",
        default="interval_hist.png",
        help="Output image path when plotting (default: interval_hist.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot in a window (may not work on headless servers).",
    )
    parser.add_argument(
        "--max-unique",
        type=int,
        default=200,
        help="If unique interval values exceed this, use binned histogram instead.",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use log scale on y-axis for the histogram.",
    )
    args = parser.parse_args()

    ts, used_path = _load_timestamp(args.h5, args.dataset_path)

    ts = np.asarray(ts).reshape(-1)
    intervals = np.diff(ts.astype(np.float64, copy=False))

    print(f"h5: {args.h5}")
    print(f"timestamp dataset: {used_path}")
    print(f"timestamp dtype: {ts.dtype}, shape: {ts.shape}")
    print("")
    print("interval summary (adjacent diffs of timestamp):")
    print(_summarize_intervals(intervals))

    if args.print_first_n and intervals.size:
        n = min(args.print_first_n, intervals.size)
        print("")
        print(f"first {n} intervals:")
        for i in range(n):
            print(f"{i}: {intervals[i]}")

    if args.plot:
        title = "Interval counts (adjacent diffs of timestamp)"
        out_path = args.out if args.out else None
        _plot_interval_histogram(
            intervals=intervals,
            title=title,
            out_path=out_path,
            show=args.show,
            max_unique=args.max_unique,
            logy=args.logy,
        )
        if out_path:
            print("")
            print(f"saved plot: {out_path}")


if __name__ == "__main__":
    main()
