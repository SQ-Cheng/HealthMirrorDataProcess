"""Extract only the verified ViT-B checkpoint from the user-provided remote ZIP."""

import json
import shutil

from . import config


URL = "https://facephyspublicmodel.tos-cn-beijing.volces.com/DINOV3.zip"
FILENAME = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
MEMBER = "DINOV3/DINOV3-pth/"+FILENAME
WEIGHTS = config.WEIGHTS.parent / FILENAME


def download():
    if WEIGHTS.exists():
        digest = config.reference.sha256(WEIGHTS)
        if not digest.startswith("73cec8be"):
            raise RuntimeError("Existing ViT-B checkpoint hash mismatch")
        print(f"[weights-reuse] {WEIGHTS} sha256={digest}", flush=True)
        return
    from remotezip import RemoteZip
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    temporary = WEIGHTS.with_suffix(".download")
    try:
        with RemoteZip(URL, timeout=120) as archive:
            info = archive.getinfo(MEMBER)
            if info.file_size != 342860279:
                raise ValueError("Unexpected official ViT-B checkpoint size")
            print(f"[download] HTTP Range: {MEMBER}; {info.file_size/2**20:.2f} MiB; no full ZIP", flush=True)
            with archive.open(info) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8*1024*1024)
        digest = config.reference.sha256(temporary)
        if temporary.stat().st_size != info.file_size or not digest.startswith("73cec8be"):
            raise RuntimeError("Downloaded checkpoint size/SHA256 mismatch")
        temporary.replace(WEIGHTS)
        audit = {"architecture": "dinov3_vitb16", "source_url": URL, "zip_member": MEMBER,
                 "size_bytes": info.file_size, "zip_crc32": f"{info.CRC:08x}", "sha256": digest,
                 "source_revision": config.REPO_REVISION, "download_method": "remote ZIP HTTP Range, one member only"}
        WEIGHTS.with_suffix(".source.json").write_text(json.dumps(audit, indent=2)+"\n")
        print(f"[weights-ready] {WEIGHTS} sha256={digest}", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    download()
