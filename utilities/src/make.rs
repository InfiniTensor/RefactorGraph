use crate::proj_dir;
use std::{collections::HashSet, ffi::OsStr, fs, path::PathBuf, process::Command};

#[derive(PartialEq, Eq, Hash, Debug)]
enum Target {
    Nvidia,
    Baidu,
}

pub fn make(
    release: bool,
    install_python: bool,
    dev: Option<Vec<String>>,
    cxx_compiler: Option<PathBuf>,
) {
    let release = if release { "Release" } else { "Debug" };
    let dev = if let Some(dev) = dev {
        dev.into_iter()
            .map(|d| d.to_ascii_lowercase())
            .filter_map(|d| match d.as_str() {
                "cuda" | "nvidia" => Some(Target::Nvidia),
                "kunlun" | "kunlunxin" | "baidu" => Some(Target::Baidu),
                _ => {
                    eprintln!("Unknown device: {}", d);
                    None
                }
            })
            .collect::<HashSet<_>>()
    } else {
        let mut ans = HashSet::new();

        println!("Checking CUDA support...");
        println!("------------------------");
        if Command::new("nvcc")
            .arg("--version")
            .status()
            .unwrap()
            .success()
        {
            ans.insert(Target::Nvidia);
            println!("------------------------");
            println!("CUDA support detected.");
        } else {
            println!("------------------------");
            println!("No CUDA support detected.");
        }
        println!();

        println!("Checking Kunlunxin support...");
        if std::env::var("KUNLUN_HOME").is_ok() {
            ans.insert(Target::Baidu);
            println!("Kunlunxin support detected.");
        } else {
            println!("No Kunlunxin support detected.");
        }
        println!();

        ans
    };
    let dev = |d: Target| if dev.contains(&d) { "ON" } else { "OFF" };

    let proj_dir = proj_dir();
    let build = proj_dir.join("build");
    fs::create_dir_all(&build).unwrap();

    {
        let mut cmake = Command::new("cmake");
        cmake
            .current_dir(&proj_dir)
            .arg("-Bbuild")
            .arg(format!("-DCMAKE_BUILD_TYPE={release}"))
            .arg(format!("-DUSE_CUDA={}", dev(Target::Nvidia)))
            .arg(format!("-DUSE_KUNLUN={}", dev(Target::Baidu)));
        if let Some(cxx_compiler) = cxx_compiler {
            cmake.arg(format!("-DCMAKE_CXX_COMPILER={}", cxx_compiler.display()));
        }
        assert!(cmake.status().unwrap().success());
    }
    {
        let mut make = Command::new("make");
        make.current_dir(&build).arg("-j");
        assert!(make.status().unwrap().success());
    }

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
