use crate::proj_dir;
use std::{ffi::OsStr, fs, path::Path, process::Command};

const SCRIPT: &str = r#"
import numpy as np
from functools import reduce
from onnxruntime import InferenceSession
from onnx import load
from refactor_graph.onnx import make_compiler

compiler = make_compiler(
    load(model_path.__str__(), load_external_data=False),
    model_path.parent.__str__(),
)
executor = compiler.compile("cuda", "default", [])

inputs = compiler.zero_inputs()
for i, input in enumerate(inputs):
    if input.dtype in [np.int64, np.int32]:
        input[...] = np.random.randint(
            0, reduce(lambda x, y: x * y, input.shape), input.shape
        ).astype(input.dtype)
    elif input.dtype in [np.float32, np.float64]:
        input[...] = np.random.random(input.shape)
    else:
        raise NotImplementedError
    executor.set_input(i, input)

executor.run()

session = InferenceSession(model_path)
answer = session.run(
    None,
    {session.get_inputs()[i].name: input for i, input in enumerate(inputs)},
)
for i, ans in enumerate(answer):
    print((executor.get_output(i) - ans).flatten())
"#;

fn model_path(path: impl AsRef<Path>) -> String {
    format!(
        "\
from pathlib import Path
model_path = Path(\"{}\").resolve()
",
        fs::canonicalize(path).unwrap().display(),
    )
}

pub fn infer(path: impl AsRef<Path>) {
    let path = path.as_ref();
    assert!(
        path.is_file() && path.extension() == Some(OsStr::new("onnx")),
        "\"{}\" is not a onnx file",
        path.display(),
    );
    Command::new("python")
        .current_dir(proj_dir())
        .arg("-c")
        .arg(format!("{}{}", model_path(path), SCRIPT))
        .status()
        .unwrap();
}
