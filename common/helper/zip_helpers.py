import logging
from pathlib import Path
import zipfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def zip_images(dataset_dir: Path, zip_name: str) -> Path | None:
    """Finds all images in dataset_dir and packs them into a local ZIP."""

    images = (
        list(dataset_dir.rglob("*.jpg"))
        + list(dataset_dir.rglob("*.jpeg"))
    )

    if not images:
        logger.warning("No images found in %s", dataset_dir)
        return None

    logger.info("Found %d images, zipping...", len(images))

    zip_path = Path("/tmp") / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for i, path in enumerate(images, 1):
            # arcname = filename only, so the zip does not include
            # kagglehub's deep cache folder structure
            zf.write(path, arcname=path.name)

            if i % 1000 == 0 or i == len(images):
                logger.info("[%d/%d] Added %s to zip", i, len(images), path.name)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info("Zip created: %s (%.1f MB)", zip_path, size_mb)

    return zip_path

def zip_dir(source_dir: Path, zip_name: str = None) -> Path:
    if zip_name is None:
        zip_path = source_dir.parent / f"{source_dir.name}.zip"
    else:
        zip_path = source_dir.parent / f"{zip_name}.zip"
    logger.info("Zipping %s → %s", source_dir, zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in source_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(source_dir))

    logger.info("Zip created (%.1f MB)", zip_path.stat().st_size / 1024 / 1024)
    return zip_path

def unzip_file(zip_path: Path, extract_to: Path = None) -> Path:
    if extract_to is None:
        extract_to = zip_path.parent / zip_path.stem
    
    logger.info("Extracting %s -> %s", zip_path, extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    logger.info("Extraction complete")

    return extract_to