mod infer;
mod make;

use clap::{Parser, Subcommand};
use std::{
    ffi::OsString,
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

fn main() {
    let proj_dir = Path::new(std::env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    match Cli::parse().command {
        Commands::Make {
            release,
            install_python,
            dev,
            cxx_compiler,
        } => make::make(proj_dir, release, install_python, dev, cxx_compiler),

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

        Commands::Infer { path } => infer::infer(proj_dir, path),
    }
}
