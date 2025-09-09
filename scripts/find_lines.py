from pathlib import Path
import sys


def find_line(path: Path, needle: str) -> int:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return i
    return -1


if __name__ == "__main__":
    p = Path(sys.argv[1])
    needle = sys.argv[2]
    print(find_line(p, needle))

