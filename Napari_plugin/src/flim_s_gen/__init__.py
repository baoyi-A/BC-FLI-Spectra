try:
    from ._version import version as __version__
except ImportError:
    try:
        from importlib.metadata import version as _ilm_version, PackageNotFoundError
        try:
            __version__ = _ilm_version("bc-flim-spectra")
        except PackageNotFoundError:
            __version__ = "unknown"
    except ImportError:
        __version__ = "unknown"

# Enable Python faulthandler so that a future Qt fast-fail / segfault /
# stack overrun dumps the full C-Python stack into a file BEFORE the
# process dies. Without this the OS just kills python.exe and we get no
# Python-side traceback at all (the user sees "napari closed silently").
# Lab-machine context: PyQt5 5.15.2 + napari Labels paintbrush has been
# observed to fast-fail (0xc0000409 STATUS_STACK_BUFFER_OVERRUN at
# Qt5Core.dll+0x204e8) on bandwidth-constrained Win11 boxes — when it
# happens again, the file below will show exactly which Python frame
# triggered it.
try:
    import faulthandler as _fh
    import os as _fh_os
    import tempfile as _fh_tempfile
    if not _fh.is_enabled():
        # Append-mode so consecutive crashes accumulate. Lives in temp
        # because anywhere under the install tree would be read-only on a
        # shared lab machine.
        _fh_log_path = _fh_os.path.join(
            _fh_tempfile.gettempdir(), 'bcflim_faulthandler.log')
        _fh_log = open(_fh_log_path, 'a', buffering=1)
        _fh.enable(file=_fh_log, all_threads=True)
except Exception:
    pass

from ._reader import napari_get_reader
from ._sample_data import make_sample_data
from ._widget import (
    PTUReader,
    BarcodeSeg,
    Calculate_FLIM_S,
    # UMAP_Class,
    SeededKMeans,
    KMeansCluster,  # backward-compat alias for SeededKMeans
    BiosensorSeg,
    Trackrevise,
    # MultiModelTracker,
    BPTracker,
)
from ._writer import write_multiple, write_single_image

# __all__ = (
#     "napari_get_reader",
#     "write_single_image",
#     "write_multiple",
#     "make_sample_data",
#     "Calculate_FLIM_S",
#     # "UMAP_Class",
#     "Trackrevise",
#     # "KMeansCluster",
#     "MultiModelTracker",
#
# )
