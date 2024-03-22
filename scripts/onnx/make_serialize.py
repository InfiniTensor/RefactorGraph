from refactor_graph.onnx import make_compiler
from onnx import load
import argparse
from onnx.external_data_helper import load_external_data_for_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Refactor compiler, export model serialize."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the model file file."
    )
    parser.add_argument(
        "--output", type=str, default="./", help="Path to save the output file."
    )
    args = parser.parse_args()
    return (
        args.model,
        args.output,
    )


def main():
    model_path, output_path = parse_args()
    model = load(model_path)
    # model = load(model_path, load_external_data=False)
    # load_external_data_for_model(
    #     model,
    #     "/home/zhangyunze/workspace/RefactorGraph/scripts/onnx/bert_bs1.pb",
    # )
    compiler = make_compiler(model)
    compiler.serialize(output_path)


if __name__ == "__main__":
    main()
