import argparse
import onnx
import numpy as np
import onnxruntime
import os
import shutil
from onnx.shape_inference import infer_shapes

input_dir_name = "inputs/"
onnx_result_dir_name = "onnx_outputs/"
size_threshold = 1024 * 1024 * 1024


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


def gen_inputs(model, working_path):
    inputs_info = model.graph.input
    dir_path = create_dir(working_path, input_dir_name)
    inputs = {}
    initializers = set(i.name for i in model.graph.initializer)

    for i, info in enumerate(inputs_info):
        if info.name not in initializers:
            shape = [
                d.dim_value if d.HasField("dim_value") else 1
                for d in info.type.tensor_type.shape.dim
            ]

            if info.type.tensor_type.elem_type == 6:
                data = np.random.randint(0, 2, size=shape).astype(np.int32)
            elif info.type.tensor_type.elem_type == 7:
                data = np.random.randint(0, 2, size=shape)
            else:
                data = np.random.random(size=shape).astype(np.float32)
            inputs[info.name] = data
            np.save(os.path.join(dir_path, f"input_{i}"), data)

    return inputs


def load_inputs(model, working_path):
    inputs_info = model.graph.input
    dir_path = os.path.join(working_path, input_dir_name)
    inputs = {}
    for i, info in enumerate(inputs_info):
        file_path = os.path.join(dir_path, f"input_{i}.npy")
        inputs[info.name] = np.load(file_path)
    return inputs


def create_dir(working_path, dir_name):
    dir_path = os.path.join(working_path, dir_name)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)

    return dir_path


def get_extra_output_groups(model):
    output_groups = []
    group = []
    total_size = 0
    for value_info in model.graph.value_info:
        group.append(value_info)
        shape = [
            d.dim_value if d.HasField("dim_value") else 1
            for d in value_info.type.tensor_type.shape.dim
        ]
        size = 1
        for i in shape:
            size *= i
        total_size += size

        if total_size > size_threshold:
            output_groups.append(group)
            group = []
            total_size = 0
    if total_size != 0:
        output_groups.append(group)
    output_groups.append([out for out in model.graph.output])
    return output_groups


def run_with_extra_outputs(model, extra_outputs, inputs, output_path):
    n = len(model.graph.output)
    for _ in range(n):
        model.graph.output.remove(model.graph.output[0])
    model.graph.output.extend(extra_outputs)
    output_names = [info.name for info in extra_outputs]
    session = onnxruntime.InferenceSession(model.SerializeToString())
    tensors = session.run(output_names=output_names, input_feed=inputs)

    for output_name, tensor in zip(output_names, tensors):
        output_name = output_name.replace("/", "_")
        output_name = output_name.replace(".", "-")
        file_path = os.path.join(output_path, output_name)
        print("Save output to " + file_path)
        np.save(file_path, tensor)


def main():
    model_path, working_path, gen_input = parse_args()
    create_dir(working_path, onnx_result_dir_name)
    model = onnx.load(model_path)
    model = infer_shapes(model)
    if gen_input:
        inputs = gen_inputs(model, working_path)
    else:
        inputs = load_inputs(model, working_path)

    output_groups = get_extra_output_groups(model)
    output_path = os.path.join(working_path, onnx_result_dir_name)

    for extra_outputs in output_groups:
        run_with_extra_outputs(model, extra_outputs, inputs, output_path)


if __name__ == "__main__":
    main()
