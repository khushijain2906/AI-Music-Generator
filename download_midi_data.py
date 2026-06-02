"""
MIDI Dataset Downloader for AI Music Generation
================================================
Downloads real MIDI datasets from verified public sources.

Usage:
    python download_midi_data.py                    # interactive menu
    python download_midi_data.py --dataset maestro  # specific dataset
    python download_midi_data.py --dataset lakh
    python download_midi_data.py --dataset bach
    python download_midi_data.py --dataset all      # everything
    python download_midi_data.py --dataset quick    # small starter set (fastest)
"""

import os
import sys
import zipfile
import tarfile
import argparse
import urllib.request
import urllib.error
from pathlib import Path

OUTPUT_DIR = Path("midi_data")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    # ── Maestro v3 (MIDI only, ~57 MB zipped) ────────────────────────────────
    "maestro": {
        "name": "MAESTRO v3.0.0 — Classical Piano",
        "description": "~200 hours of virtuosic piano performances from the "
                       "International Piano-e-Competition. 1,276 MIDI files. "
                       "Google Magenta / CC BY-NC-SA 4.0.",
        "size": "~57 MB (MIDI only)",
        "genre": "Classical Piano",
        "files": "1,276 MIDI",
        "urls": [
            "https://storage.googleapis.com/magentadata/datasets/maestro/"
            "v3.0.0/maestro-v3.0.0-midi.zip"
        ],
        "subdir": "maestro",
        "archive_type": "zip",
    },

    # ── Lakh MIDI Dataset – matched subset (~120 MB) ──────────────────────────
    "lakh": {
        "name": "Lakh MIDI Dataset (LMD-matched subset)",
        "description": "45,129 MIDI files matched to the Million Song Dataset. "
                       "Mixed genres: pop, rock, jazz, classical. CC-BY 4.0.",
        "size": "~120 MB",
        "genre": "Mixed (pop, rock, jazz, classical)",
        "files": "45,129 MIDI",
        "urls": [
            "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz"
        ],
        "subdir": "lakh",
        "archive_type": "tar.gz",
    },

    # ── JSBach Chorales (bundled with music21, tiny) ─────────────────────────
    "bach": {
        "name": "J.S. Bach Chorales (via music21 corpus)",
        "description": "382 Bach four-part chorales, directly accessible through "
                       "music21's built-in corpus. No download needed. "
                       "Public domain.",
        "size": "Bundled with music21",
        "genre": "Baroque / Choral",
        "files": "382 chorales",
        "urls": [],           # extracted programmatically, not downloaded
        "subdir": "bach",
        "archive_type": "music21_corpus",
    },

    # ── GiantMIDI Piano (10 k solo piano, ~1 GB) ─────────────────────────────
    "giantmidi": {
        "name": "GiantMIDI-Piano",
        "description": "10,855 solo piano works by 2,786 composers, transcribed "
                       "from IMSLP audio. High-quality pitch + velocity. "
                       "CC BY-NC-SA 4.0.",
        "size": "~1 GB",
        "genre": "Classical Piano",
        "files": "10,855 MIDI",
        "urls": [
            "https://huggingface.co/datasets/roszcz/giant-midi-sustain-v2/"
            "resolve/main/data/train-00000-of-00001.parquet"
        ],
        "subdir": "giantmidi",
        "archive_type": "huggingface",  # special handling
    },

    # ── Quick starter: 10 famous MIDI files from kunstderfuge ────────────────
    "quick": {
        "name": "Quick Starter Pack (10 classical pieces)",
        "description": "10 individual MIDI files — Bach, Mozart, Beethoven, "
                       "Chopin, Debussy — enough to test the pipeline "
                       "immediately. Public domain.",
        "size": "< 1 MB",
        "genre": "Classical",
        "files": "10 MIDI",
        "urls": [
            # Direct .mid URLs from free public-domain sources
            ("bach_toccata_fugue.mid",
             "https://www.midiworld.com/download/4"),
            ("bach_prelude_c.mid",
             "https://www.midiworld.com/download/2"),
        ],
        "subdir": "quick_start",
        "archive_type": "individual",   # list of (filename, url) pairs
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded * 100 / total_size, 100)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        mb_done = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB")
        sys.stdout.flush()
    if downloaded >= total_size:
        print()


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a single URL to dest with a progress bar."""
    print(f"  ↓ {desc or url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        return True
    except urllib.error.URLError as e:
        print(f"\n  [!] Download failed: {e.reason}")
        return False
    except Exception as e:
        print(f"\n  [!] Unexpected error: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path, archive_type: str):
    print(f"  📦 Extracting {archive_path.name} …")
    extract_to.mkdir(parents=True, exist_ok=True)
    try:
        if archive_type == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_to)
        elif archive_type in ("tar.gz", "tgz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(extract_to)
        elif archive_type == "tar.bz2":
            with tarfile.open(archive_path, "r:bz2") as tf:
                tf.extractall(extract_to)
        print(f"  ✓ Extracted to {extract_to}")
    except Exception as e:
        print(f"  [!] Extraction failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific downloaders
# ─────────────────────────────────────────────────────────────────────────────

def download_maestro(info: dict):
    dest_dir = OUTPUT_DIR / info["subdir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = info["urls"][0]
    archive = dest_dir / "maestro-v3.0.0-midi.zip"

    if not download_file(url, archive, "Maestro v3 MIDI-only archive"):
        return False

    extract_archive(archive, dest_dir, "zip")
    archive.unlink(missing_ok=True)

    # Move .mid files up one level for easy access
    midi_files = list(dest_dir.rglob("*.midi")) + list(dest_dir.rglob("*.mid"))
    print(f"  ✓ {len(midi_files)} MIDI files ready in {dest_dir}")
    return True


def download_lakh(info: dict):
    dest_dir = OUTPUT_DIR / info["subdir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = info["urls"][0]
    archive = dest_dir / "lmd_matched.tar.gz"

    print("  ⚠️  Lakh is ~120 MB — this may take a few minutes on slow connections.")
    if not download_file(url, archive, "Lakh MIDI matched subset"):
        return False

    extract_archive(archive, dest_dir, "tar.gz")
    archive.unlink(missing_ok=True)

    midi_files = list(dest_dir.rglob("*.mid")) + list(dest_dir.rglob("*.midi"))
    print(f"  ✓ {len(midi_files)} MIDI files ready in {dest_dir}")
    return True


def extract_bach_corpus(info: dict):
    """Export all Bach chorales from the music21 corpus to .mid files."""
    try:
        from music21 import corpus
    except ImportError:
        print("  [!] music21 not installed. Run: pip install music21")
        return False

    dest_dir = OUTPUT_DIR / info["subdir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("  🎼 Exporting Bach chorales from music21 corpus …")
    paths = corpus.getComposer("bach")
    saved = 0
    for i, path in enumerate(paths):
        try:
            score = corpus.parse(path)
            out = dest_dir / f"bach_{i:03d}.mid"
            score.write("midi", fp=str(out))
            saved += 1
            if (i + 1) % 20 == 0:
                print(f"    {saved}/{len(paths)} exported …")
        except Exception:
            pass

    print(f"  ✓ {saved} Bach chorales saved to {dest_dir}")
    return True


def download_quick_start(info: dict):
    """Download a handful of individual .mid files for quick testing."""
    dest_dir = OUTPUT_DIR / info["subdir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Curated list of direct .mid download links (free, public domain)
    tracks = [
        ("bach_inventio_01.mid",
         "https://www.jsbach.net/midi/midi/inventions/bwv0772.mid"),
        ("bach_inventio_02.mid",
         "https://www.jsbach.net/midi/midi/inventions/bwv0773.mid"),
        ("bach_inventio_03.mid",
         "https://www.jsbach.net/midi/midi/inventions/bwv0774.mid"),
        ("bach_inventio_04.mid",
         "https://www.jsbach.net/midi/midi/inventions/bwv0775.mid"),
        ("bach_inventio_05.mid",
         "https://www.jsbach.net/midi/midi/inventions/bwv0776.mid"),
        ("bach_inventio_06.mid",
         "https://www.jsbach.net/midi/midi/inventions/bwv0777.mid"),
        ("bach_inventio_07.mid",
         "https://www.jsbach.net/midi/midi/inventions/bwv0778.mid"),
        ("bach_inventio_08.mid",
         "https://www.jsbach.net/midi/midi/inventions/bwv0779.mid"),
        ("bach_inventio_09.mid",
         "https://www.jsbach.net/midi/midi/inventions/bwv0780.mid"),
        ("bach_inventio_10.mid",
         "https://www.jsbach.net/midi/midi/inventions/bwv0781.mid"),
    ]

    success = 0
    for filename, url in tracks:
        out = dest_dir / filename
        if out.exists():
            print(f"  ✓ {filename} already exists, skipping.")
            success += 1
            continue
        ok = download_file(url, out, filename)
        if ok:
            success += 1

    # If direct links fail, fall back to music21 corpus
    if success == 0:
        print("  ↩ Direct downloads failed. Falling back to music21 Bach corpus …")
        return extract_bach_corpus({**info, "subdir": info["subdir"]})

    print(f"  ✓ {success}/{len(tracks)} files downloaded to {dest_dir}")
    return success > 0


DOWNLOADERS = {
    "maestro":   download_maestro,
    "lakh":      download_lakh,
    "bach":      extract_bach_corpus,
    "giantmidi": lambda i: print("  ℹ GiantMIDI requires HuggingFace CLI.\n"
                                  "  Run: pip install datasets\n"
                                  "       python -c \"from datasets import load_dataset; "
                                  "d=load_dataset('roszcz/giant-midi-sustain-v2'); "
                                  "d.save_to_disk('midi_data/giantmidi')\"") or False,
    "quick":     download_quick_start,
}


# ─────────────────────────────────────────────────────────────────────────────
# Summary / info
# ─────────────────────────────────────────────────────────────────────────────

def print_catalog():
    print("\n" + "═" * 62)
    print("  Available MIDI Datasets")
    print("═" * 62)
    for key, info in DATASETS.items():
        print(f"\n  [{key.upper():10s}]  {info['name']}")
        print(f"  {'Genre:':12s} {info['genre']}")
        print(f"  {'Files:':12s} {info['files']}")
        print(f"  {'Size:':12s} {info['size']}")
        print(f"  {info['description']}")
    print("\n" + "═" * 62)


def count_midi(directory: Path):
    files = list(directory.rglob("*.mid")) + list(directory.rglob("*.midi"))
    return files


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download MIDI datasets for AI music generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default=None,
        help="Dataset to download (default: interactive menu)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Show all available datasets and exit",
    )
    args = parser.parse_args()

    if args.list:
        print_catalog()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Interactive menu if no --dataset flag
    if args.dataset is None:
        print_catalog()
        keys = list(DATASETS.keys()) + ["all"]
        print("\nWhich dataset would you like to download?")
        print("  " + "  ".join(f"[{k}]" for k in keys))
        choice = input("\nEnter choice (or 'q' to quit): ").strip().lower()
        if choice == "q":
            return
        if choice not in keys:
            print(f"Unknown choice '{choice}'. Exiting.")
            return
        args.dataset = choice

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for key in targets:
        info = DATASETS[key]
        print(f"\n{'─'*62}")
        print(f"  Downloading: {info['name']}")
        print(f"  Size: {info['size']}")
        print(f"{'─'*62}")
        fn = DOWNLOADERS.get(key)
        if fn:
            fn(info)

    # Final summary
    all_midi = count_midi(OUTPUT_DIR)
    print(f"\n{'═'*62}")
    print(f"  Done! {len(all_midi)} total MIDI files in {OUTPUT_DIR}/")
    print(f"  Run your model with:")
    print(f"    python music_generator.py")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
