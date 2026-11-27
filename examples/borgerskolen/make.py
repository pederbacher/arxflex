# Setup
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path.cwd()
ROOT = HERE.parent.parent
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(
    [str(ROOT / "src"), str(HERE / "src")] + [os.environ.get("PYTHONPATH", "")]))


def run(*cmd):
    """Run a command in this folder; raises an error if it fails."""
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, env=ENV, check=True)


# Render the vignette, reusing the cached cells
run("quarto", "render", "vignette.py")

# Render with every cell run again in a new kernel: after changing arxflex
# (the cache only notices changed cells, and Quarto's kernel still has the old
# arxflex imported)
run("quarto", "render", "vignette.py", "--cache-refresh", "--execute-daemon-restart")

# Live preview in the browser, re-rendered on save; runs in the background
preview = subprocess.Popen(["quarto", "preview", "vignette.py"], cwd=HERE, env=ENV)

# Stop the preview
preview.terminate()

# Executed notebook vignette.ipynb (needs: pip install jupytext)
run(sys.executable, "-m", "jupytext", "--to", "ipynb", "--execute", "vignette.py")

# check_fitmodel.py: the new code against the legacy models
run(sys.executable, "check_fitmodel.py")

# Build both wheels and list their contents
# The core must not contain the example data, and neither may contain the
# legacy code or the scripts of this folder. setuptools reuses its copy in
# build/, so a file deleted from src/ would still end up in the wheel.
for build in (ROOT / "build", HERE / "build"):
    shutil.rmtree(build, ignore_errors=True)
wheel_dir = tempfile.mkdtemp()
run(sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
    "-q", "-w", wheel_dir, str(ROOT), str(HERE))
for wheel in sorted(Path(wheel_dir).glob("*.whl")):
    print(f"\n{wheel.name}")
    for name in zipfile.ZipFile(wheel).namelist():
        if ".dist-info/" not in name:
            print("   ", name)
shutil.rmtree(wheel_dir)

# Clean: remove the rendered page, its cache, the notebook and the temporary
# notebooks (vignette.quarto_ipynb_1, ...) an interrupted render leaves behind
for path in [HERE / name for name in ("vignette.html", "vignette_files", ".jupyter_cache",
                                      ".quarto", "vignette.ipynb")] \
            + sorted(HERE.glob("vignette.quarto_ipynb*")):
    name = path.name
    if path.is_dir():
        shutil.rmtree(path)
        print("removed", name)
    elif path.exists():
        path.unlink()
        print("removed", name)
