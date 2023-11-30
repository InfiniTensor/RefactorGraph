import argparse
import numpy as np
import os
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Compare output files.")
    parser.add_argument(
        "--expect", type=str, default="./onnx_outputs/", help="Path to the expected output files by onnx_runtime."
    )
    parser.add_argument(
        "--actual", type=str, default="./if_outputs/", help="Path to the actual output files."
    )
    args = parser.parse_args()
    print("arg setting: ", args)
    return (
        args.expect,
        args.actual,
    )

def compare_npy(actual_path, expect_path, edge, node):
    actual = np.load(actual_path)
    expect = np.load(expect_path)
    if np.isnan(actual).any():
        print(f"NAN value in node:{node} edge:{edge}")
        return
    
    result = np.allclose(actual, expect, rtol= 1e-3, atol=1e-3)
    if not result:
        print(f"Failed at node:{node} edge:{edge}")
    else:
        print(f"Pass node:{node} edge:{edge}")

def main():
    expect_dir, actual_dir = parse_args()
    actual_files = [f for f in os.listdir(actual_dir) if f.endswith('.npy')]

    for actual_file in actual_files:
        name_group = re.search(r'([^()]+)(\([^()]+\))\.npy', actual_file)
        if name_group:
            edge_name = name_group.group(1)
            node_name = name_group.group(2)[1:-1]

            expect_file = edge_name + ".npy"
            expect_file_path = os.path.join(expect_dir, expect_file)
            if os.path.exists(expect_file_path):
                actual_file_path = os.path.join(actual_dir, actual_file)
                compare_npy(actual_file_path, expect_file_path, edge_name, node_name)
            else:
                print("Could not find output file for " + expect_file)
        else:
            print("Invalid file name pattern: " + "actaul_file")



if __name__ == "__main__":
    main()
