import argparse
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Compare output files.")
    parser.add_argument(
        "--expect",
        type=str,
        default="./onnx_outputs/",
        help="Path to the expected output files by onnx_runtime.",
    )
    parser.add_argument(
        "--actual",
        type=str,
        default="./if_outputs/",
        help="Path to the actual output files.",
    )
    args = parser.parse_args()
    return (
        args.expect,
        args.actual,
    )

def getDiff(base, test):
    absolute_diff = np.subtract(base, test)
    max_absolute_diff = np.max(np.abs(absolute_diff))

    baseCopy = base.astype(np.float64).ravel()
    testCopy = test.astype(np.float64).ravel()
    upValue = np.sum(np.abs(baseCopy - testCopy))
    downValue = np.sum(np.abs(baseCopy)) + np.float64(1e-9)
    max_relative_diff = upValue / downValue

    return max_absolute_diff, max_relative_diff

def compare_npy(actual_path, expect_path, edge, node):
    actual = np.load(actual_path)
    expect = np.load(expect_path)
    if np.isnan(actual).any():
        print(f"NAN value in node:{node} edge:{edge}")
        return
    
    max_absolute_diff, max_relative_diff = getDiff(expect, actual)
    if max_absolute_diff != 0.0: ## No need to print tensor with no diff
        print(f'{max_absolute_diff}\t{max_relative_diff}\t{node}\t{edge}')


def main():
    expect_dir, actual_dir = parse_args()
    meta_files = sorted([f for f in os.listdir(actual_dir) if f.endswith(".meta")])
    for meta_file in meta_files:
        with open(os.path.join(actual_dir, meta_file), "r") as file:
            node_name = ""
            for line in file:
                elements = line.strip().split()
                if "node" == elements[0]:
                    node_id, node_name = elements[1], elements[2]
                elif ("input" == elements[0] or "output" == elements[0]) and len(
                    elements
                ) == 4:
                    edge_id, edge_name, actual_file_path = (
                        elements[1],
                        elements[2],
                        elements[3],
                    )
                    expect_file = edge_name.replace("/", "_")
                    expect_file = expect_file.replace(".", "-")
                    expect_file = expect_file + ".npy"
                    expect_file_path = os.path.join(expect_dir, expect_file)
                    if os.path.exists(expect_file_path):
                        compare_npy(
                            actual_file_path, expect_file_path, edge_name, node_name
                        )


if __name__ == "__main__":
    main()
