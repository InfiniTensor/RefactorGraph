mod format;
mod infer;
mod make;

use clap::{Parser, Subcommand};
use std::{
    ffi::OsString,
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
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
    /// format source files
    Format,
    /// run model inference
    Infer { path: PathBuf },
}

pub fn proj_dir() -> &'static Path {
    static PROJ: OnceLock<&Path> = OnceLock::new();
    *PROJ.get_or_init(|| Path::new(std::env!("CARGO_MANIFEST_DIR")).parent().unwrap())
}

fn main() {
    match Cli::parse().command {
        Commands::Make {
            release,
            install_python,
            dev,
            cxx_compiler,
        } => make::make(release, install_python, dev, cxx_compiler),

        Commands::Clean => match fs::remove_dir_all(proj_dir().join("build")) {
            Ok(_) => {}
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(e) => panic!("{}", e),
        },

        Commands::Test => {
            Command::new("make")
                .current_dir(proj_dir().join("build"))
                .arg("test")
                .arg("-j")
                .status()
                .unwrap();
        }

        Commands::Format => format::format(),

        Commands::Infer { path } => infer::infer(path),
    }
}
