"""
Adaptive Variable-Resolution 2.5D LiDAR Mapping
Main Application Entry Point.
"""
import argparse
import os
import sys
import webbrowser
import uvicorn
import yaml

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from benchmarking.runner import BenchmarkRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception"
    )
    parser.add_argument("--demo", action="store_true", help="Launch interactive 3D WebGL Dashboard Demo")
    parser.add_argument("--benchmark", action="store_true", help="Run comprehensive offline benchmark suite")
    parser.add_argument("--input", type=str, default=None, help="Path to LiDAR data folder or .bin/.pcd/.ply file")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser on demo start")
    return parser.parse_args()


def run_benchmark(config_path: str, input_path: str = None):
    print("==========================================================================")
    print(" [*] RUNNING ADAPTIVE 2.5D LiDAR MAPPING BENCHMARK SUITE")
    print("==========================================================================")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    runner = BenchmarkRunner(config)
    report = runner.run_full_benchmark(num_frames=20, source_path=input_path)

    summary = report["summary"]
    b = report["uniform_baseline_5cm"]
    d = report["adaptive_foveated_default"]
    a = report["adaptive_foveated_aggressive"]

    print(f"\n[SUMMARY] Evaluated {summary['frames_evaluated']} frames | Total Points: {summary['total_points_evaluated']:,} | Avg Points/Frame: {summary['avg_points_per_frame']:,}")
    print("-" * 74)
    print(f"{'Configuration':<30} | {'Avg Cells':<10} | {'Mem (KB)':<10} | {'Latency':<10} | {'Savings':<8}")
    print("-" * 74)
    print(f"{'Uniform 5cm Baseline':<30} | {b['avg_cells']:<10} | {b['avg_memory_kb']:<10} | {b['avg_insertion_latency_ms']:<7.2f} ms | Baseline")
    print(f"{'Adaptive Default (5-60cm)':<30} | {d['avg_cells']:<10} | {d['avg_memory_kb']:<10} | {d['avg_insertion_latency_ms']:<7.2f} ms | -{d['cell_reduction_pct']}%")
    print(f"{'Adaptive Aggressive (5-80cm)':<30} | {a['avg_cells']:<10} | {a['avg_memory_kb']:<10} | {a['avg_insertion_latency_ms']:<7.2f} ms | -{a['cell_reduction_pct']}%")
    print("-" * 74)
    print(f"[+] Default Speedup Factor: {d['speedup_factor']}x | Aggressive Speedup: {a['speedup_factor']}x\n")


def run_server(args):
    print("==========================================================================")
    print(" [>] ADAPTIVE VARIABLE-RESOLUTION 2.5D LiDAR MAPPING SYSTEM")
    print(" Smart India Hackathon (SIH) Real-Time Perception Prototype")
    print("==========================================================================")
    print(f"[*] Loading config from: {args.config}")
    print(f"[*] Starting WebGL Dashboard on http://{args.host}:{args.port}")

    from visualization.server import create_app
    app = create_app(config_path=args.config, source_path=args.input)

    if not args.no_browser:
        def open_browser():
            import time
            time.sleep(1.2)
            webbrowser.open(f"http://{args.host}:{args.port}")
        import threading
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def main():
    args = parse_args()
    if args.benchmark:
        run_benchmark(args.config, args.input)
    else:
        run_server(args)


if __name__ == "__main__":
    main()
