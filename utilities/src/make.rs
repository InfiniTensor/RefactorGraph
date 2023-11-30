use std::{
    collections::HashSet,
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(PartialEq, Eq, Hash, Debug)]
enum Target {
    Nvidia,
    Baidu,
}

pub fn make(
    proj_dir: impl AsRef<Path>,
    release: bool,
    install_python: bool,
    dev: Option<Vec<OsString>>,
    cxx_compiler: Option<PathBuf>,
) {
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

    let build = proj_dir.as_ref().join("build");
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
            .as_ref()
            .join("src/09python_ffi/src/refactor_graph")
            .join(from.file_name().unwrap());
        fs::copy(from, to).unwrap();

        Command::new("pip")
            .arg("install")
            .arg("-e")
            .arg(proj_dir.as_ref().join("src/09python_ffi/"))
            .status()
            .unwrap();
    }
}
