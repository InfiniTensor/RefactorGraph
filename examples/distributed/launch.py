import argparse
import os
import time
import multiprocessing as mp
from refactor_graph.onnx import make_compiler, find_device
import onnx
from onnx.external_data_helper import convert_model_to_external_data
from onnx.shape_inference import infer_shapes_path
import numpy as np
from parallel_opt import parallel_model


os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="launch distributed infinitensor")
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")
    parser.add_argument(
        "--nproc_per_node", type=int, default=1, help="number of processes per node"
    )
    parser.add_argument(
        "--name", type=str, default="test", help="name of this instance."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="path to the ONNX model file."
    )
    parser.add_argument(
        "--gen_std",
        action="store_true",
        help="whether to generate random inputs and standard results, should be set when running for the first time.",
    )
    args = parser.parse_args()
    print("arg setting: ", args)
    return (
        args.num_nodes,
        args.nproc_per_node,
        args.name,
        args.model,
        args.gen_std,
    )


def run_model(executor, inputs, n=10):
    for i in range(len(inputs)):
        executor.set_input(i, inputs[i])
    
    executor.run()
    # get outputs
    outputs = executor.get_output(0)

    # bench
    if n > 0:
        begin = time.time()
        for _ in range(n):
            executor.run()
        end = time.time()
        avg_time = (end - begin) / n
        print(f"average time: {avg_time}")
    return outputs


def run_and_compare(name, executor, inputs):

    results = np.load(f"{name}_results.npy")
    outputs = run_model(executor, inputs, 10)
    print("outputs abs mean:", abs(outputs).mean())
    np.testing.assert_allclose(outputs, results, rtol=1e-3, atol=1e-5)

def load_inputs(name, compiler):
    inputs = compiler.zero_inputs()
    for i, input in enumerate(inputs):
        input[...] = np.load(f"{name}_input_{i}.npy")
    return inputs

def start_worker(
    name: str, world_size: int, rank: int, local_rank: int, model: onnx.ModelProto
):
    dist_name = name + "_dist"
    model = parallel_model(model, world_size, rank)
    extern_path = f"./{dist_name}_rank{rank}.pb"
    if os.path.exists(extern_path):
        os.remove(extern_path)
    onnx.save_model(
        model,
        f"./{dist_name}_rank{rank}.onnx",
        save_as_external_data=True,
        location=extern_path,
    )
    infer_shapes_path(f"./{dist_name}_rank{rank}.onnx")
    
    compiler = make_compiler(model, ".")

    inputs = load_inputs(name, compiler)
    
    executor = compiler.compile_on(find_device("nvidia", rank), "default", [])
    executor.set_cuda_commnication(world_size, rank)

    run_and_compare(name, executor, inputs)


def start_single(name, model, gen_input):
    compiler = make_compiler(model)
    if gen_input:
        inputs = compiler.zero_inputs()
        for i, input in enumerate(inputs):
            input[...] = np.random.random(input.shape).astype(input.dtype)
            np.save(f"{name}_input_{i}", input)
    else:
        inputs = load_inputs(name, compiler)
    executor = compiler.compile_on(find_device("nvidia", 0), "default", [])
    outputs = run_model(executor, inputs, 10)
    print("outputs abs mean:", abs(outputs).mean())
    np.save(f"{name}_results", outputs)


def main():
    nnodes, nproc_per_node, name, model_path, gen_input = parse_args()

    model = onnx.load(model_path)

    # run single process.
    # use standalone process to isolate cuda.
    print("run model by single GPU.")
    p = mp.Process(target=start_single, args=(name, model, gen_input))
    p.start()
    p.join()

    # run distributed parallel.
    world_size = nnodes * nproc_per_node
    print(f"run model by {world_size} GPU in parallel.")
    workers = [
        mp.Process(
            target=start_worker,
            args=(name, world_size, rank, rank % nproc_per_node, model),
        )
        for rank in range(world_size)
    ]

    for w in workers:
        w.start()

    for w in workers:
        w.join()


if __name__ == "__main__":
    main()
