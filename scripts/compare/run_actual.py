import numpy as np
from pathlib import Path
from onnx import load
from refactor_graph.onnx import make_compiler
import argparse
import os
import shutil

input_dir_name = "inputs/"
result_dir_name = "if_outputs/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run with onnx_runtime, export all outputs to file."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the ONNX model file."
    )
    parser.add_argument("--output", type=str, default="./", help="Working directory.")

    parser.add_argument(
        "--gen_input", action="store_true", help="Generate random input."
    )
    args = parser.parse_args()
    print("arg setting: ", args)
    return (
        args.model,
        args.output,
        args.gen_input,
    )


def create_dir(working_path, dir_name):
    dir_path = os.path.join(working_path, dir_name)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)

    return dir_path


def main():
    model_path, work_path, gen_input = parse_args()
    if gen_input:
        create_dir(work_path, input_dir_name)
    create_dir(work_path, result_dir_name)

    model = load(model_path, load_external_data=False)
    compiler = make_compiler(model, Path(model_path).parent.__str__())
    executor = compiler.compile("cuda", "default", [])
    inputs = compiler.zero_inputs()

    for i, input in enumerate(inputs):
        if gen_input:
            input[...] = np.random.random(input.shape).astype(input.dtype)
            executor.set_input(i, input)
            np.save(os.path.join(work_path, input_dir_name, f"input_{i}"), input)
        else:
            input = np.load(os.path.join(work_path, input_dir_name, f"input_{i}.npy"))
            executor.set_input(i, input)
    executor.trace(os.path.join(work_path, result_dir_name), "npy")


if __name__ == "__main__":
    main()
