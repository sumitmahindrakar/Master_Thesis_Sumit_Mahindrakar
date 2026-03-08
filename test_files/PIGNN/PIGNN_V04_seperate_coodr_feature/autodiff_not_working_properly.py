from pathlib import Path

print(Path.cwd())

from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
print(CURRENT_SUBFOLDER)