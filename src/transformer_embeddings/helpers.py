from os.path import commonpath
from pathlib import Path
from tarfile import open as tarfile_open
from typing import List, Optional, Union


def compress_files(
    filenames: List[Union[str, Path]],
    compressed_file: Union[str, Path],
    arcname: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Given a list of files or directories, compress them into a tarball at the
    compressed_file location.

    Parameters
    ----------
    filenames : List[Union[str, Path]]
        List of files or directories
    compressed_file : Union[str, Path]
        Destination compressed file.
    arcname : Union[str, Path]
        Base directory for the compressed file. Default: Maximum common path for all
        the passed in filenames.

    Returns
    -------
    bool
        The status of the compress operation. Checks if the output file was created.

    Raises
    ------
    NameError
        Raised when the expected name of the compressed file doesn't end in tar or tar.gz.
    """
    # If the file to be written is provided as an str object, convert it to a Path object.
    if isinstance(compressed_file, str):
        compressed_file = Path(compressed_file)

    # Ensure the file to be written ends with tar or tar.gz.
    if compressed_file.name.endswith(".tar.gz") or compressed_file.name.endswith(
        ".tar"
    ):
        tar_file = tarfile_open(compressed_file, "w:gz")
    else:
        raise NameError(
            f"Name of compressed file ({compressed_file}) does not end in tar or tar.gz."
        )

    base_path = Path(commonpath(filenames))

    for filename in filenames:
        if isinstance(filename, str):
            filename = Path(filename)

        if arcname is None:
            arcname = filename.relative_to(base_path)
            # If common path is "", the arcname becomes a directory `.` in which all
            # files are then stored. This line avoids that scenario.
            arcname = "" if arcname.as_posix() == "." else arcname
        tar_file.add(filename, arcname=arcname)
    tar_file.close()
    return compressed_file.exists()
