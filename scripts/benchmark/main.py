import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.benchmark.methods.algorithms import dijkstra_shortest_path, astar_shortest_path
from scripts.benchmark.methods.benchmark import run_full_benchmark

def main():
    results = run_full_benchmark()
    return results


if __name__ == '__main__':
    main()
