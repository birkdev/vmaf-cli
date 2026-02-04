# vmaf-cli
A CLI tool for calculating [VMAF](https://github.com/Netflix/vmaf) scores between original/distorted video files. Features parallel segment processing, rich terminal UI, RAM disk support, and JSON export.

## Why use this?

VMAF is the industry standard metric for measuring perceptual video quality, developed by Netflix. Running it manually with ffmpeg or the vmaf CLI involves a lot of repetitive command-line work. This tool automates the entire workflow — segmenting, extracting, scoring, and reporting — so you can compare encodes quickly without writing one-off scripts.

## Requirements

- **Python 3.8+**
- **[rich](https://github.com/Textualize/rich)** — installed automatically by the launcher, or manually with `pip install rich`
- **[FFmpeg](https://ffmpeg.org/)** — `ffmpeg` and `ffprobe` must be in your PATH
- **[vmaf](https://github.com/Netflix/vmaf)** — the standalone `vmaf` CLI must be in your PATH

> **Note:** RAM disk support (via [ImDisk](http://www.ltr-data.se/opencode.html/#ImDisk)) is only available on Windows and requires administrator privileges. On other platforms, the tool falls back to a temporary directory automatically.

## Getting started

### Using the PowerShell launcher (Windows)

```powershell
.\vmaf_calc.ps1
```

This checks for Python and the `rich` package, installs it if needed, and launches the calculator. You can also pass arguments:

```powershell
.\vmaf_calc.ps1 -Original "C:\videos\ref.mkv" -Distorted "C:\videos\encoded.mkv" -Workers 4
```

### Running the Python script directly

```bash
python vmaf_modern.py
```

Or with command-line arguments to skip the interactive prompts:

```bash
python vmaf_modern.py --original ref.mkv --distorted encoded.mkv --segment-duration 30 --model standard --workers 2
```

Any omitted option will be prompted interactively.

### CLI options

| Option | Description |
|---|---|
| `--original PATH` | Path to the original (reference) video |
| `--distorted PATH` | Path to the distorted (encoded) video |
| `--segment-duration N` | Segment duration in seconds (default: 30) |
| `--model MODEL` | VMAF model: `standard` or `4k` (default: standard) |
| `--workers N` | Parallel worker count (default: 2) |
| `--help` | Show help message |
