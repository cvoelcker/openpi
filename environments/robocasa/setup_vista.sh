set -euxo pipefail

BASE="$SCRATCH/openpi/environments/robocasa"
VENV="$BASE/.venv"
CACHE_ROOT="$BASE/.cache"

cd "$BASE"

UV_BIN="$(command -v uv)"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_PYTHON_INSTALL_DIR="$CACHE_ROOT/python"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

export UV_PROJECT_ENVIRONMENT="$VENV"

mkdir -p \
    "$UV_CACHE_DIR" \
    "$UV_PYTHON_INSTALL_DIR" \
    "$XDG_CACHE_HOME"

# evdev fails when its native extension is built with NVIDIA nvc.
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

date --iso-8601=seconds
hostname
uname -a

echo "BASE=$BASE"
echo "VENV=$VENV"
echo "UV_BIN=$UV_BIN"

"$UV_BIN" --version
"$CC" --version | head -1
"$CXX" --version | head -1
df -h "$BASE"

grep -A8 '^dependencies' "$BASE/pyproject.toml"

"$UV_BIN" python install 3.11

"$UV_BIN" sync \
    --project "$BASE" \
    --frozen \
    --python 3.11

test -x "$VENV/bin/python"

"$UV_BIN" pip show \
    --python "$VENV/bin/python" \
    robocasa

"$UV_BIN" pip show \
    --python "$VENV/bin/python" \
    robosuite

"$VENV/bin/python" - <<'PY'
import mujoco
import numpy
import robocasa
import robosuite

print("IMPORT_OK")
print("numpy", numpy.__version__)
print("mujoco", mujoco.__version__)
print("robocasa", robocasa.__file__)
print("robosuite", robosuite.__file__)
PY

"$UV_BIN" run \
    --project "$BASE" \
    --frozen \
    python -m robocasa.scripts.setup_macros

ASSET_SENTINEL="$VENV/lib/python3.11/site-packages/robocasa/models/assets/fixtures/stoves/Stove028/model.xml"

if [[ -f "$ASSET_SENTINEL" ]]; then
    echo "RoboCasa kitchen assets already exist."
else
    printf 'y\n' |
        "$UV_BIN" run \
            --project "$BASE" \
            --frozen \
            python -m robocasa.scripts.download_kitchen_assets
fi

test -f "$ASSET_SENTINEL"


"$UV_BIN" run \
    --project "$BASE" \
    --frozen \
    python "$BASE/test.py"

date --iso-8601=seconds
echo "ROBOCASA_SETUP_COMPLETE"
