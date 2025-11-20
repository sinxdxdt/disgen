"""Dataset stubs with installation instructions.

MUSDB18-HQ:
    pip install musdb
    # Download dataset from https://zenodo.org/record/3338373

MoisesDB:
    pip install moisesdb
    # See https://github.com/moises-ai/moisesdb

Slakh2100:
    # Download from http://www.slakh.com/
    # Manual implementation required
"""

class MUSDB18HQStub:
    """Stub for MUSDB18-HQ dataset.
    
    To implement MUSDB18-HQ:
    1. Install: pip install musdb
    2. Download dataset from https://zenodo.org/record/3338373
    3. Implement BaseDataset interface
    4. Register with @DatasetRegistry.register("musdb18hq")
    
    Dataset structure:
    - 100 train tracks
    - 50 test tracks
    - 4 stems: vocals, drums, bass, other
    - Sample rate: 44100 Hz
    - Stereo audio
    """
    pass


class MoisesDBStub:
    """Stub for MoisesDB dataset.
    
    To implement MoisesDB:
    1. Install: pip install moisesdb
    2. Use moisesdb Python API to access data
    3. Implement BaseDataset interface
    4. Register with @DatasetRegistry.register("moisesdb")
    
    Dataset structure:
    - 240 tracks total
    - No explicit test split (use sampling)
    - Hierarchical stem structure
    - Multiple genres
    """
    pass


class Slakh2100Stub:
    """Stub for Slakh2100 dataset.
    
    To implement Slakh2100:
    1. Download from http://www.slakh.com/
    2. Use slakh_utils if available
    3. Implement BaseDataset interface
    4. Register with @DatasetRegistry.register("slakh2100")
    
    Dataset structure:
    - 1500 train tracks
    - 375 validation tracks
    - 225 test tracks
    - Variable number of stems per track
    - Sample rate: 44100 Hz
    - Mono audio in FLAC format
    """
    pass
