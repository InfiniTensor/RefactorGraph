use std::{
    ffi::OsStr,
    fs,
    os::unix::ffi::OsStrExt,
    path::Path,
    process::{Child, Command},
};

use crate::proj_dir;

pub fn format() {
    for mut ele in recur(&proj_dir().join("src")) {
        let status = ele.wait().unwrap();
        if !status.success() {
            println!("{:?}", status);
        }
    }
}

fn recur(path: impl AsRef<Path>) -> Vec<Child> {
    fs::read_dir(path)
        .unwrap()
        .into_iter()
        .filter_map(|entry| entry.ok())
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                if path.ends_with("09python_ffi/pybind11") {
                    vec![]
                } else {
                    recur(&path)
                }
            } else if let Some(child) = format_one(&path) {
                vec![child]
            } else {
                vec![]
            }
        })
        .flatten()
        .collect::<Vec<_>>()
}

fn format_one(file: &Path) -> Option<Child> {
    const C_STYLE_FILE: [&[u8]; 9] = [
        b"h", b"hh", b"hpp", b"c", b"cc", b"cpp", b"cxx", b"cu", b"mlu",
    ];
    let Some(ext) = file.extension() else {
        return None;
    };
    if C_STYLE_FILE.contains(&ext.as_bytes()) {
        Command::new("clang-format-14")
            .arg("-i")
            .arg(file)
            .spawn()
            .ok()
    } else if ext == OsStr::new("py") {
        Command::new("black").arg(file).spawn().ok()
    } else {
        None
    }
}

// 根据 git diff 判断格式化哪些文件的功能，暂时没用
// if len(sys.argv) == 1:
//     # Last commit.
//     print("Formats git added files.")
//     for line in (
//         run("git status", cwd=proj_path, capture_output=True, shell=True)
//         .stdout.decode()
//         .splitlines()
//     ):
//         line = line.strip()
//         # Only formats git added files.
//         for pre in ["new file:", "modified:"]:
//             if line.startswith(pre):
//                 format_file(line[len(pre) :].strip())
//             break
// else:
//     # Origin commit.
//     origin = sys.argv[1]
//     print(f'Formats changed files from "{origin}".')
//     for line in (
//         run(f"git diff {origin}", cwd=proj_path, capture_output=True, shell=True)
//         .stdout.decode()
//         .splitlines()
//     ):
//         diff = "diff --git "
//         if line.startswith(diff):
//             files = line[len(diff) :].split(" ")
//             assert len(files) == 2
//             assert files[0][:2] == "a/"
//             assert files[1][:2] == "b/"
//             format_file(files[1][2:])
