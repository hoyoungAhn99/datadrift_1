from __future__ import annotations

import argparse

from core.result_export import export_result_to_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    created = export_result_to_csv(args.result, args.output_dir)
    for path in created:
        print(path)


if __name__ == "__main__":
    main()

