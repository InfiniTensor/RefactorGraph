from refactor_graph.onnx import make_compiler
from onnx import load
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Refactor compiler, export model serialize."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the model file file."
    )
    parser.add_argument("--output", type=str, default="./", help="Path to save the output file.")
    args = parser.parse_args()
    return (
        args.model,
        args.output,
    )

def main():
    model_path, output_path = parse_args()
    compiler = make_compiler(load(model_path))
    compiler.serialize(output_path)

if __name__ == "__main__":
    main()