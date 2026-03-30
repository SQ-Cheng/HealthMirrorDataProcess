from data.copy_data import CopyFiles
import shutil

mirror_id = input("Enter mirror ID (e.g., 1, 2): ").strip()

copyfiles = CopyFiles(
    source_dir=f"./mirror{mirror_id}_data",
    dest_dir=f"./mirror{mirror_id}_download",
    filenames=[
        "rppg_log.csv",
    ],
    log=True
)

copyfiles.copy()

archive_name = f"mirror{mirror_id}_download"
shutil.make_archive(archive_name, 'zip', copyfiles.dest_dir)
