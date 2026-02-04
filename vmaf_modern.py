#!/usr/bin/env python3
"""
Modern VMAF Video Quality Calculator
Requires: pip install rich
"""

import os
import sys
import time
import subprocess
import xml.etree.ElementTree as ET
import json
import shutil
import statistics
import tempfile
import atexit
import ctypes
import platform
from pathlib import Path
from datetime import timedelta
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn,
        TaskProgressColumn
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.rule import Rule
    from rich import box
except ImportError:
    print("Error: Required package 'rich' not found.")
    print("Please install it with: pip install rich")
    sys.exit(1)

console = Console()


def parse_args() -> dict:
    """Parse command-line arguments. Missing args fall through to interactive prompts."""
    args: dict = {
        'original': None,
        'distorted': None,
        'segment_duration': None,
        'model': None,
        'workers': None,
    }
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--help':
            console.print(Panel.fit(
                "[bold cyan]VMAF Video Quality Calculator[/bold cyan]\n"
                "[dim]Modern CLI Edition[/dim]",
                border_style="bright_blue", box=box.DOUBLE
            ))
            console.print()
            console.print("[bold]Usage:[/bold]  python vmaf_modern.py [OPTIONS]")
            console.print()
            tbl = Table(show_header=True, box=box.SIMPLE)
            tbl.add_column("Option", style="cyan")
            tbl.add_column("Description")
            tbl.add_row("--original PATH", "Path to the original (reference) video")
            tbl.add_row("--distorted PATH", "Path to the distorted (encoded) video")
            tbl.add_row("--segment-duration N", "Segment duration in seconds (default: 30)")
            tbl.add_row("--model MODEL", "VMAF model: 'standard' or '4k' (default: standard)")
            tbl.add_row("--workers N", "Parallel worker count (default: 2)")
            tbl.add_row("--help", "Show this help message")
            console.print(tbl)
            console.print()
            console.print("[dim]Any omitted option will be prompted interactively.[/dim]")
            sys.exit(0)
        elif argv[i] == '--original' and i + 1 < len(argv):
            args['original'] = argv[i + 1]; i += 2
        elif argv[i] == '--distorted' and i + 1 < len(argv):
            args['distorted'] = argv[i + 1]; i += 2
        elif argv[i] == '--segment-duration' and i + 1 < len(argv):
            args['segment_duration'] = int(argv[i + 1]); i += 2
        elif argv[i] == '--workers' and i + 1 < len(argv):
            args['workers'] = int(argv[i + 1]); i += 2
        elif argv[i] == '--model' and i + 1 < len(argv):
            args['model'] = argv[i + 1].lower(); i += 2
        else:
            console.print(f"[yellow]Unknown argument: {argv[i]}[/yellow]")
            i += 1
    return args


class RamDiskManager:
    """Manages RAM disk creation and cleanup using ImDisk"""

    def __init__(self, drive_letter: str = "R:", size: str = "25G"):
        self.drive_letter = drive_letter.rstrip(':') + ':'
        self.size = size
        self.created = False
        self.imdisk_path = self._find_imdisk()

    def _find_imdisk(self) -> Optional[str]:
        """Find ImDisk executable"""
        possible_paths = [
            r"C:\Windows\System32\imdisk.exe",
            r"C:\Program Files\ImDisk\imdisk.exe",
            r"C:\Program Files (x86)\ImDisk\imdisk.exe",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        try:
            result = subprocess.run(["where", "imdisk"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except FileNotFoundError:
            pass

        return None

    def is_admin(self) -> bool:
        """Check if running with administrator privileges"""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except AttributeError:
            return False

    def exists(self) -> bool:
        """Check if RAM disk already exists"""
        return os.path.exists(self.drive_letter)

    def create(self) -> bool:
        """Create RAM disk"""
        if not self.imdisk_path:
            console.print("[red]ImDisk not found! Please install ImDisk Virtual Disk Driver.[/red]")
            console.print("Download from: http://www.ltr-data.se/opencode.html/#ImDisk")
            return False

        if self.exists():
            console.print(f"[green]RAM disk {self.drive_letter} already exists[/green]")
            return True

        if not self.is_admin():
            console.print("[yellow]Creating RAM disk requires administrator privileges![/yellow]")
            console.print("Please run this script as administrator.")
            return False

        console.print(f"[cyan]Creating {self.size} RAM disk at {self.drive_letter}...[/cyan]")

        try:
            cmd = [
                self.imdisk_path,
                "-a", "-s", self.size,
                "-m", self.drive_letter,
                "-p", "/fs:ntfs /q /y"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.created = True
                console.print(f"[green]RAM disk created at {self.drive_letter}[/green]")
                return True
            else:
                console.print(f"[red]Failed to create RAM disk: {result.stderr}[/red]")
                return False

        except Exception as e:
            console.print(f"[red]Error creating RAM disk: {e}[/red]")
            return False

    def remove(self):
        """Remove RAM disk if we created it"""
        if not self.created or not self.imdisk_path:
            return
        if not self.exists():
            return

        console.print(f"[cyan]Removing RAM disk {self.drive_letter}...[/cyan]")
        try:
            cmd = [self.imdisk_path, "-D", "-m", self.drive_letter]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                console.print("[green]RAM disk removed successfully[/green]")
            else:
                console.print(f"[yellow]Warning: Could not remove RAM disk: {result.stderr}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Error removing RAM disk: {e}[/yellow]")


class VMAFCalculator:
    def __init__(self):
        self.console = console
        self.ram_disk_manager = RamDiskManager("R:", "25G")
        self.ram_disk_path = "R:\\"
        self.use_temp_dir = False
        self.temp_dir = None
        self.vmaf_threads = 32
        self.ffmpeg_path = "ffmpeg"
        self.ffprobe_path = "ffprobe"
        self.vmaf_path = "vmaf"
        self.original_video: str = ""
        self.distorted_video: str = ""
        self.segment_duration: int = 30
        self.total_duration: float = 0
        self.vmaf_model: str = "version=vmaf_v0.6.1"
        self.parallel_workers: int = 2
        self.scores: list = []
        self.segment_results: list = []  # (index, start_time, duration, score)
        self.segment_timings: list = []
        self.start_time: Optional[float] = None
        self.total_segments: int = 0

        atexit.register(self.cleanup)

    def cleanup(self):
        """Cleanup RAM disk and temporary files"""
        if self.ram_disk_manager:
            self.ram_disk_manager.remove()
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except OSError:
                pass

    def get_working_directory(self) -> str:
        """Get the working directory for temporary files"""
        if self.use_temp_dir:
            if not self.temp_dir:
                self.temp_dir = tempfile.mkdtemp(prefix="vmaf_")
            return self.temp_dir
        return self.ram_disk_path

    def check_tools(self) -> bool:
        """Validate that required tools are available before proceeding"""
        tools = {
            'ffmpeg': self.ffmpeg_path,
            'ffprobe': self.ffprobe_path,
            'vmaf': self.vmaf_path,
        }
        all_ok = True
        for name, cmd in tools.items():
            try:
                result = subprocess.run(
                    [cmd, "-version" if name != "vmaf" else "--version"],
                    capture_output=True, text=True
                )
                version_line = result.stdout.strip().split('\n')[0] if result.stdout.strip() else "found"
                self.console.print(f"  [green]✓[/green] {name}: {version_line}")
            except FileNotFoundError:
                self.console.print(f"  [red]✗[/red] {name}: NOT FOUND")
                all_ok = False
        return all_ok

    def get_file_path(self, prompt_text: str, file_type: str) -> str:
        """Get file path with validation"""
        while True:
            self.console.print(f"[yellow]{prompt_text}[/yellow]")
            path = Prompt.ask("Path", console=self.console)
            path = path.strip('"').strip("'")

            if not path:
                self.console.print("[red]Path cannot be empty![/red]")
                continue
            if not os.path.isfile(path):
                self.console.print(f"[red]File not found: {path}[/red]")
                continue

            self.console.print(f"[green]✓ {file_type} set to:[/green] {path}")
            return path

    def validate_video(self, path: str) -> bool:
        """Check that the file has at least one video stream"""
        try:
            result = subprocess.run(
                [self.ffprobe_path, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_type",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True
            )
            return "video" in result.stdout.strip()
        except Exception:
            return False

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe"""
        try:
            result = subprocess.run(
                [self.ffprobe_path, "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                 video_path],
                capture_output=True, text=True, check=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            self.console.print(f"[red]Error getting video duration: {e}[/red]")
            return 0

    def format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS.mmm"""
        td = timedelta(seconds=seconds)
        total = td.total_seconds()
        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        secs = total % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def process_segment(self, segment_data: dict) -> Tuple[int, float, float, Optional[float], float]:
        """Process a single segment. Returns (index, start_time, duration, vmaf_score, time_taken)"""
        segment_idx = segment_data['index']
        start_time = segment_data['start_time']
        duration = segment_data['duration']

        segment_start = time.time()
        working_dir = self.get_working_directory()

        original_y4m = os.path.join(working_dir, f"original_segment_{segment_idx}.y4m")
        distorted_y4m = os.path.join(working_dir, f"distorted_segment_{segment_idx}.y4m")
        vmaf_json = os.path.join(working_dir, f"vmaf_output_{segment_idx}.json")
        vmaf_xml = os.path.join(working_dir, f"vmaf_output_{segment_idx}.xml")
        temp_files = [original_y4m, distorted_y4m, vmaf_json, vmaf_xml]

        try:
            # Extract both segments in parallel using Popen
            proc_orig = subprocess.Popen([
                self.ffmpeg_path, "-hide_banner", "-loglevel", "error",
                "-ss", str(start_time), "-t", str(duration),
                "-i", self.original_video, "-pix_fmt", "yuv420p",
                "-fps_mode", "cfr", "-an", "-y", original_y4m
            ])
            proc_dist = subprocess.Popen([
                self.ffmpeg_path, "-hide_banner", "-loglevel", "error",
                "-ss", str(start_time), "-t", str(duration),
                "-i", self.distorted_video, "-pix_fmt", "yuv420p",
                "-fps_mode", "cfr", "-an", "-y", distorted_y4m
            ])
            proc_orig.wait()
            proc_dist.wait()

            if proc_orig.returncode != 0 or proc_dist.returncode != 0:
                raise RuntimeError("ffmpeg extraction failed")

            # Try JSON output first, fall back to XML
            score = None
            try:
                subprocess.run([
                    self.vmaf_path, "-r", original_y4m, "-d", distorted_y4m,
                    "--threads", str(self.vmaf_threads), "--model", self.vmaf_model,
                    "--json", "-o", vmaf_json
                ], check=True, capture_output=True)

                with open(vmaf_json, 'r') as f:
                    data = json.load(f)
                # pooled_metrics.vmaf.mean or aggregate level
                pooled = data.get("pooled_metrics", {})
                if "vmaf" in pooled:
                    score = pooled["vmaf"].get("mean")
                if score is None:
                    # Try frames-level average
                    frames = data.get("frames", [])
                    if frames:
                        vmaf_vals = [fr["metrics"]["vmaf"] for fr in frames if "vmaf" in fr.get("metrics", {})]
                        if vmaf_vals:
                            score = sum(vmaf_vals) / len(vmaf_vals)
            except Exception:
                # Fall back to XML
                try:
                    subprocess.run([
                        self.vmaf_path, "-r", original_y4m, "-d", distorted_y4m,
                        "--threads", str(self.vmaf_threads), "--model", self.vmaf_model,
                        "--xml", "-o", vmaf_xml
                    ], check=True, capture_output=True)

                    tree = ET.parse(vmaf_xml)
                    root = tree.getroot()
                    for path in [".//aggregate_metrics",
                                 ".//metric[@name='vmaf']",
                                 ".//{http://www.spirent.com/ns/vmaf}metric[@name='vmaf']"]:
                        elem = root.find(path)
                        if elem is not None:
                            score = float(elem.get('mean', elem.get('VMAF_score', '0')))
                            if score > 0:
                                break
                except Exception:
                    pass

            # Cleanup temp files
            for f in temp_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

            return segment_idx, start_time, duration, score, time.time() - segment_start

        except Exception:
            for f in temp_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            return segment_idx, start_time, duration, None, time.time() - segment_start

    def update_stats_panel(self) -> Panel:
        """Create statistics panel"""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        elapsed = time.time() - self.start_time if self.start_time else 0

        table.add_row("Elapsed Time", self.format_time(elapsed))
        table.add_row("Segments Processed", f"{len(self.scores)} / {self.total_segments}")
        table.add_row("Parallel Workers", str(self.parallel_workers))
        table.add_row("Working Directory", self.get_working_directory())

        if self.scores:
            avg_score = sum(self.scores) / len(self.scores)
            table.add_row("Current Average VMAF", f"{avg_score:.2f}")
            table.add_row("Score Range", f"{min(self.scores):.2f} - {max(self.scores):.2f}")

        if len(self.segment_timings) > 1:
            recent = self.segment_timings[-5:]
            avg_time = sum(recent) / len(recent)
            table.add_row("Avg Time per Segment", f"{avg_time:.1f}s")
            remaining = self.total_segments - len(self.scores)
            # Account for parallelism in ETA
            eta_seconds = (remaining * avg_time) / self.parallel_workers
            table.add_row("Estimated Remaining", self.format_time(eta_seconds))

        return Panel(table, title="[bold]Statistics[/bold]", border_style="green")

    def run_analysis(self):
        """Run the VMAF analysis with modern UI"""
        self.start_time = time.time()

        segments = []
        current_time = 0.0
        for i in range(self.total_segments):
            seg_duration = min(self.segment_duration, self.total_duration - current_time)
            if seg_duration < 1:
                break
            segments.append({
                'index': i + 1,
                'start_time': current_time,
                'duration': seg_duration
            })
            current_time += self.segment_duration

        self.total_segments = len(segments)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=10
        )

        storage_status = (
            f"[green]Using RAM Disk (R:)[/green]" if not self.use_temp_dir
            else "[yellow]Using Temp Directory[/yellow]"
        )

        header_panel = Panel(
            f"[bold cyan]Processing {self.total_segments} segments "
            f"({self.parallel_workers} workers)[/bold cyan]\n{storage_status}",
            box=box.ROUNDED
        )

        stats_panel = self.update_stats_panel()
        current_panel = Panel("[dim]Starting...[/dim]", border_style="blue")
        main_task = progress.add_task("[cyan]Overall Progress", total=self.total_segments)

        def create_display():
            display = Table.grid(padding=1)
            display.add_column(justify="center", width=console.width)
            display.add_row(header_panel)
            display.add_row(Panel(progress, border_style="blue", title="Progress"))
            display.add_row(stats_panel)
            display.add_row(current_panel)
            return display

        with Live(create_display(), console=self.console, refresh_per_second=10) as live:
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                futures = {executor.submit(self.process_segment, seg): seg for seg in segments}

                for future in as_completed(futures):
                    seg_idx, seg_start, seg_dur, score, time_taken = future.result()

                    if score is not None:
                        self.scores.append(score)
                        self.segment_results.append((seg_idx, seg_start, seg_dur, score))
                        self.segment_timings.append(time_taken)
                        current_panel = Panel(
                            f"[green]✓[/green] Segment {seg_idx}: VMAF = {score:.2f}  ({time_taken:.1f}s)",
                            border_style="green"
                        )
                    else:
                        self.segment_results.append((seg_idx, seg_start, seg_dur, None))
                        current_panel = Panel(
                            f"[red]✗[/red] Segment {seg_idx}: Failed",
                            border_style="red"
                        )

                    progress.update(main_task, advance=1)
                    stats_panel = self.update_stats_panel()
                    live.update(create_display())

        # Sort results by segment index for ordered display
        self.segment_results.sort(key=lambda x: x[0])

    def display_results(self):
        """Display final results with style"""
        self.console.print()
        self.console.rule("[bold cyan]Analysis Complete[/bold cyan]", style="cyan")
        self.console.print()

        if not self.scores:
            self.console.print("[red]No VMAF scores were collected![/red]")
            return

        avg_score = sum(self.scores) / len(self.scores)
        min_score = min(self.scores)
        max_score = max(self.scores)

        # Summary table
        table = Table(title="VMAF Analysis Results", box=box.ROUNDED,
                      title_style="bold cyan", header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="white", width=20)
        table.add_column("Assessment", style="yellow", width=30)

        table.add_row("Segments Analyzed", f"{len(self.scores)} / {self.total_segments}", "")
        table.add_row("Average VMAF Score", f"{avg_score:.4f}", self.get_quality_assessment(avg_score))
        table.add_row("Minimum Score", f"{min_score:.2f}", "")
        table.add_row("Maximum Score", f"{max_score:.2f}", "")

        if len(self.scores) > 1:
            std_dev = statistics.stdev(self.scores)
            table.add_row("Standard Deviation", f"{std_dev:.4f}", "")

        total_time = time.time() - self.start_time
        table.add_row("Total Processing Time", self.format_time(total_time), "")

        if self.segment_timings:
            avg_seg_time = sum(self.segment_timings) / len(self.segment_timings)
            table.add_row("Avg Time per Segment", f"{avg_seg_time:.1f}s", "")

        storage_method = "RAM Disk (R:)" if not self.use_temp_dir else "Temporary Directory"
        table.add_row("Storage Method", storage_method,
                      "[green]Fast[/green]" if not self.use_temp_dir else "[yellow]Standard[/yellow]")
        table.add_row("Parallel Workers", str(self.parallel_workers), "")

        self.console.print(table)

        # Per-segment breakdown table
        self.console.print()
        seg_table = Table(title="Per-Segment Scores", box=box.SIMPLE_HEAVY,
                          title_style="bold cyan", header_style="bold")
        seg_table.add_column("#", style="dim", width=4, justify="right")
        seg_table.add_column("Time Range", width=25)
        seg_table.add_column("VMAF", justify="right", width=10)
        seg_table.add_column("Rating", width=20)

        for idx, seg_start, seg_dur, score in self.segment_results:
            time_range = f"{self.format_time(seg_start)} - {self.format_time(seg_start + seg_dur)}"
            if score is not None:
                rating = self.get_quality_assessment(score)
                seg_table.add_row(str(idx), time_range, f"{score:.2f}", rating)
            else:
                seg_table.add_row(str(idx), time_range, "[red]FAILED[/red]", "")

        self.console.print(seg_table)

        # Visual score bar
        self.console.print()
        self.draw_score_bar(avg_score)

        # Offer to export results
        self.console.print()
        if Confirm.ask("[yellow]Export results to JSON?[/yellow]", default=False, console=self.console):
            self.export_results(avg_score, min_score, max_score, total_time)

    def export_results(self, avg_score, min_score, max_score, total_time):
        """Save results to a JSON file next to the distorted video"""
        try:
            distorted_path = Path(self.distorted_video)
            export_path = distorted_path.with_suffix('.vmaf_results.json')

            results = {
                'original': self.original_video,
                'distorted': self.distorted_video,
                'model': self.vmaf_model,
                'segment_duration': self.segment_duration,
                'total_duration': self.total_duration,
                'parallel_workers': self.parallel_workers,
                'average_vmaf': round(avg_score, 4),
                'min_vmaf': round(min_score, 4),
                'max_vmaf': round(max_score, 4),
                'std_dev': round(statistics.stdev(self.scores), 4) if len(self.scores) > 1 else 0,
                'total_time_seconds': round(total_time, 2),
                'segments': [
                    {
                        'index': idx,
                        'start': round(seg_start, 3),
                        'duration': round(seg_dur, 3),
                        'vmaf': round(score, 4) if score is not None else None
                    }
                    for idx, seg_start, seg_dur, score in self.segment_results
                ]
            }

            with open(export_path, 'w') as f:
                json.dump(results, f, indent=2)

            self.console.print(f"\n[green]Results exported to:[/green] {export_path}")
        except Exception as e:
            self.console.print(f"[yellow]Could not export results: {e}[/yellow]")

    def get_quality_assessment(self, score: float) -> str:
        """Get quality assessment for VMAF score"""
        if score >= 95:
            return "[green]Excellent - Imperceptible loss[/green]"
        elif score >= 90:
            return "[green]Very Good - Barely noticeable[/green]"
        elif score >= 80:
            return "[yellow]Good - Noticeable but acceptable[/yellow]"
        elif score >= 70:
            return "[yellow]Fair - Noticeable and annoying[/yellow]"
        elif score >= 60:
            return "[red]Poor - Annoying quality loss[/red]"
        else:
            return "[red]Bad - Very annoying loss[/red]"

    def draw_score_bar(self, score: float):
        """Draw a visual score bar"""
        bar_width = 50
        filled = int((score / 100) * bar_width)

        if score >= 90:
            color = "green"
        elif score >= 70:
            color = "yellow"
        else:
            color = "red"

        bar = f"[{color}]{'\u2588' * filled}[/{color}]{'\u2591' * (bar_width - filled)}"

        self.console.print(Panel(
            f"{bar} {score:.1f}%",
            title="[bold]Average VMAF Score[/bold]",
            border_style=color
        ))

    def run(self, cli_args: Optional[dict] = None):
        """Main entry point"""
        if cli_args is None:
            cli_args = {}

        self.console.clear()
        self.console.print(Panel.fit(
            "[bold cyan]VMAF Video Quality Calculator[/bold cyan]\n"
            "[dim]Modern CLI Edition[/dim]",
            border_style="bright_blue", box=box.DOUBLE
        ))
        self.console.print()

        # Pre-flight: check required tools
        self.console.print("[bold]Checking required tools...[/bold]")
        if not self.check_tools():
            self.console.print("\n[red]Missing required tools. Please install them and ensure they are in PATH.[/red]")
            return

        self.console.print()

        # Original video
        if cli_args.get('original') and os.path.isfile(cli_args['original']):
            self.original_video = cli_args['original']
            self.console.print(f"[green]✓ Original video:[/green] {self.original_video}")
        else:
            self.original_video = self.get_file_path(
                "Enter the path to the ORIGINAL video file:", "Original video"
            )

        if not self.validate_video(self.original_video):
            self.console.print("[red]File does not appear to contain a video stream![/red]")
            return

        # Distorted video
        if cli_args.get('distorted') and os.path.isfile(cli_args['distorted']):
            self.distorted_video = cli_args['distorted']
            self.console.print(f"[green]✓ Distorted video:[/green] {self.distorted_video}")
        else:
            self.distorted_video = self.get_file_path(
                "Enter the path to the DISTORTED video file:", "Distorted video"
            )

        if not self.validate_video(self.distorted_video):
            self.console.print("[red]File does not appear to contain a video stream![/red]")
            return

        self.console.print()

        # Segment duration
        if cli_args.get('segment_duration') and cli_args['segment_duration'] >= 1:
            self.segment_duration = cli_args['segment_duration']
            self.console.print(f"[green]✓ Segment duration:[/green] {self.segment_duration}s")
        else:
            while True:
                self.segment_duration = IntPrompt.ask(
                    "[yellow]Enter segment duration in seconds[/yellow]",
                    default=30, console=self.console
                )
                if self.segment_duration >= 1:
                    break
                self.console.print("[red]Segment duration must be at least 1 second.[/red]")

        # VMAF model
        if cli_args.get('model') in ('standard', '4k'):
            choice = cli_args['model']
        else:
            self.console.print()
            model_table = Table(show_header=False, box=box.ROUNDED)
            model_table.add_column("Option", style="cyan")
            model_table.add_column("Model", style="white")
            model_table.add_column("Description", style="dim")
            model_table.add_row("1", "Standard (vmaf_v0.6.1)", "For HD content and below")
            model_table.add_row("2", "4K (vmaf_4k_v0.6.1)", "Optimized for 4K content")
            self.console.print(model_table)
            choice = Prompt.ask(
                "[yellow]Select VMAF model[/yellow]",
                choices=["1", "2"], default="1", console=self.console
            )
            choice = "standard" if choice == "1" else "4k"

        self.vmaf_model = "version=vmaf_v0.6.1" if choice == "standard" else "version=vmaf_4k_v0.6.1"

        # Workers
        if cli_args.get('workers') and cli_args['workers'] >= 1:
            self.parallel_workers = cli_args['workers']
        else:
            self.console.print()
            self.parallel_workers = IntPrompt.ask(
                "[yellow]Parallel workers (segments processed simultaneously)[/yellow]",
                default=2, console=self.console
            )
            if self.parallel_workers < 1:
                self.parallel_workers = 1

        # Setup RAM disk
        self.console.print()
        with self.console.status("[cyan]Setting up RAM disk...", spinner="dots"):
            if not self.ram_disk_manager.create():
                self.console.print("[yellow]Using temporary directory instead.[/yellow]")
                self.use_temp_dir = True
                self.console.print(f"[cyan]Working directory: {self.get_working_directory()}[/cyan]")

        # Get video duration
        self.console.print()
        with self.console.status("[cyan]Analyzing video files...", spinner="dots"):
            self.total_duration = self.get_video_duration(self.original_video)

        if self.total_duration <= 0:
            self.console.print("[red]Could not determine video duration![/red]")
            return

        self.total_segments = int(self.total_duration / self.segment_duration) + (
            1 if self.total_duration % self.segment_duration > 0 else 0
        )

        # Confirmation summary
        self.console.print()
        summary = Table(title="Settings Summary", box=box.ROUNDED,
                        title_style="bold cyan", show_header=False)
        summary.add_column("Setting", style="cyan", width=22)
        summary.add_column("Value", style="white")
        summary.add_row("Original", self.original_video)
        summary.add_row("Distorted", self.distorted_video)
        summary.add_row("Duration", self.format_time(self.total_duration))
        summary.add_row("Segment Duration", f"{self.segment_duration}s")
        summary.add_row("Total Segments", str(self.total_segments))
        summary.add_row("VMAF Model", self.vmaf_model.replace("version=", ""))
        summary.add_row("Parallel Workers", str(self.parallel_workers))
        storage = "RAM Disk (R:)" if not self.use_temp_dir else f"Temp ({self.get_working_directory()})"
        summary.add_row("Storage", storage)
        self.console.print(summary)
        self.console.print()

        if not Confirm.ask("[yellow]Start analysis?[/yellow]", default=True, console=self.console):
            self.console.print("[dim]Cancelled.[/dim]")
            return

        # Run analysis
        self.console.print()
        self.run_analysis()

        # Show results
        self.display_results()

        self.console.print()
        self.console.print("[dim]Press Enter to exit...[/dim]")
        input()


if __name__ == "__main__":
    try:
        if platform.system() != "Windows":
            console.print("[yellow]RAM disk functionality is only available on Windows.[/yellow]")
            console.print("[yellow]The script will use a temporary directory instead.[/yellow]")

        cli_args = parse_args()
        calculator = VMAFCalculator()
        calculator.run(cli_args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)
    finally:
        if 'calculator' in locals():
            calculator.cleanup()
