use clap::{Parser, Subcommand};
use std::{
    collections::HashSet,
    ffi::{OsStr, OsString},
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    process::Command,
};

/// Refactor Graph utilities
#[derive(Debug, Parser)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// make project
    Make {
        /// build type
        #[clap(long)]
        release: bool,
        /// install python frontend
        #[clap(long)]
        install_python: bool,
        /// devices support
        #[clap(long)]
        dev: Option<Vec<OsString>>,
        /// specify c++ compiler
        #[clap(long)]
        cxx_compiler: Option<PathBuf>,
    },
    /// clean build files
    Clean,
    /// run tests
    Test,
    /// run model inference
    Infer { path: PathBuf },
}

#[derive(PartialEq, Eq, Hash, Debug)]
enum Target {
    Nvidia,
    Baidu,
}

fn main() {
    let proj_dir = Path::new(std::env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    match Cli::parse().command {
        Commands::Make {
            release,
            install_python,
            dev,
            cxx_compiler,
        } => {
            let release = if release { "Release" } else { "Debug" };
            let dev = dev
                .unwrap_or_default()
                .into_iter()
                .map(|d| d.to_ascii_lowercase())
                .filter_map(|d| {
                    if d == OsStr::new("cuda") || d == OsStr::new("nvidia") {
                        Some(Target::Nvidia)
                    } else if d == OsStr::new("kunlun")
                        || d == OsStr::new("kunlunxin")
                        || d == OsStr::new("baidu")
                    {
                        Some(Target::Baidu)
                    } else {
                        eprintln!("warning: unknown device: {:?}", d);
                        None
                    }
                })
                .collect::<HashSet<_>>();
            let dev = |d: Target| if dev.contains(&d) { "ON" } else { "OFF" };

            let build = proj_dir.join("build");
            fs::create_dir_all(&build).unwrap();

            let mut cmd = Command::new("cmake");
            cmd.current_dir(&proj_dir)
                .arg("-Bbuild")
                .arg(format!("-DCMAKE_BUILD_TYPE={release}"))
                .arg(format!("-DUSE_CUDA={}", dev(Target::Nvidia)))
                .arg(format!("-DUSE_KUNLUN={}", dev(Target::Baidu)));
            if let Some(cxx_compiler) = cxx_compiler {
                cmd.arg(format!("-DCMAKE_CXX_COMPILER={}", cxx_compiler.display()));
            }
            cmd.status().unwrap();

            Command::new("make")
                .current_dir(&build)
                .arg("-j")
                .status()
                .unwrap();

            if install_python {
                let from = fs::read_dir(build.join("src/09python_ffi"))
                    .unwrap()
                    .filter_map(|ele| ele.ok())
                    .find(|entry| {
                        entry
                            .path()
                            .extension()
                            .filter(|&ext| ext == OsStr::new("so"))
                            .is_some()
                    })
                    .unwrap()
                    .path();
                let to = proj_dir
                    .join("src/09python_ffi/src/refactor_graph")
                    .join(from.file_name().unwrap());
                fs::copy(from, to).unwrap();

                Command::new("pip")
                    .arg("install")
                    .arg("-e")
                    .arg(proj_dir.join("src/09python_ffi/"))
                    .status()
                    .unwrap();
            }
        }
        Commands::Clean => match fs::remove_dir_all(proj_dir.join("build")) {
            Ok(_) => {}
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(e) => panic!("{}", e),
        },
        Commands::Test => {
            Command::new("make")
                .current_dir(proj_dir.join("build"))
                .arg("test")
                .arg("-j")
                .status()
                .unwrap();
        }
        Commands::Infer { path } => {
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
            assert!(
                path.is_file() && path.extension() == Some(OsStr::new("onnx")),
                "\"{}\" is not a onnx file",
                path.display(),
            );
            Command::new("python")
                .current_dir(proj_dir)
                .arg("-c")
                .arg(format!(
                    "\
from pathlib import Path
model_path = Path(\"{}\").resolve()
{}",
                    fs::canonicalize(path).unwrap().display(),
                    SCRIPT
                ))
                .status()
                .unwrap();
        }
    }
}
