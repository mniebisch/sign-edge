import pathlib

import click
import numpy as np
import tqdm
from skimage import color, feature, io

__all__ = ["main"]


def find_image_files(directory_path: pathlib.Path | str) -> list[pathlib.Path]:
    # Create a Path object for the directory
    directory = pathlib.Path(directory_path)

    # Define a list of image file extensions you want to search for
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]

    # Use list comprehension to find all image files with specified extensions
    image_files = [
        file for file in directory.iterdir() if file.suffix.lower() in image_extensions
    ]

    return image_files


def remove_alpha_channel(
    img: np.ndarray, fill_color: tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    if not all([0 <= channel_value <= 255 for channel_value in fill_color]):
        raise ValueError("Invalid fill color. All values must be in range [0, 255]")
    img = np.where(
        img[:, :, [3]] > 0, img[:, :, :3], np.ones_like(img[:, :, :3]) * fill_color
    )
    return img.astype(np.uint8)


@click.command()
@click.option(
    "--src",
    required=True,
    type=click.Path(exists=True, path_type=pathlib.Path, resolve_path=True),
    help="Path to directory containing images to process.",
)
@click.option(
    "--dst",
    required=True,
    type=click.Path(path_type=pathlib.Path, resolve_path=True),
    help="Path to directory to save edge images to. If directory does not exist, "
    "it will be created.",
)
@click.option(
    "--overwrite-img",
    is_flag=True,
    help="Flag to enable overwrite of existing images. Default is to skip duplicate "
    "images in dst.",
)
def main(src: pathlib.Path, dst: pathlib.Path, overwrite_img: bool) -> None:
    if not dst.exists():
        click.echo(f"DST '{dst}' does not exist. Creating it...")
        dst.mkdir(parents=True)

    img_files = find_image_files(src)

    for file in tqdm.tqdm(img_files, unit="images", desc="Processing images"):
        img = io.imread(file)
        if img.ndim != 3:
            click.echo(f"Image '{file}' is not RGB. Skipping...")
            continue

        if img.shape[2] == 4:
            img = remove_alpha_channel(img)

        img_gray = color.rgb2gray(img)
        img_edges = feature.canny(img_gray)
        output_file = dst / file.name
        if output_file.exists() and not overwrite_img:
            click.echo(f"File '{output_file}' already exists. Skipping...")
            continue

        # Invert the image to make it black background and white edges
        img_edges = np.logical_not(img_edges)
        img_edges = img_edges.astype(np.uint8) * 255

        io.imsave(output_file, img_edges, check_contrast=False)
    click.echo(f"Edge images saved to: {dst}")
