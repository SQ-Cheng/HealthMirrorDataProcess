import shutil
import os

class CopyFiles:
    def __init__(self, source_dir, dest_dir, filenames, log=False):
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.filenames = filenames
        self.log = log

    def copy(self):
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)

        for subdir in os.listdir(self.source_dir):
            source_subdir = os.path.join(self.source_dir, subdir)
            dest_subdir = os.path.join(self.dest_dir, subdir)

            if not os.path.isdir(source_subdir):
                continue

            if not os.path.exists(dest_subdir):
                os.makedirs(dest_subdir)

            for filename in self.filenames:
                source_file = os.path.join(source_subdir, filename)
                dest_file = os.path.join(dest_subdir, filename)

                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_file)
                    if self.log:
                        print(f"[CopyFiles] Copied {source_file} to {dest_file}")
                else:
                    if self.log:
                        print(f"[CopyFiles] Warning: Source file does not exist: {source_file}")

