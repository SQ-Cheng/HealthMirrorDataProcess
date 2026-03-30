from data.copy_data import CopyFiles

copyfiles = CopyFiles(
    source_dir="./mirror1_data",
    dest_dir="./mirror1_upload",
    filenames=[
        "video.avi",
        "video.avi.ts",
    ],
    log=True
)

copyfiles.copy()
