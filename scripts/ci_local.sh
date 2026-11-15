set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
WITH_DOCKER=0
FAST=0
for arg in "$@"; do
    case "$arg" in
        --with-docker) WITH_DOCKER=1 ;;
        --fast)        FAST=1 ;;
        -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

VENV="${VENV:-venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        VENV=".venv"
    else
        cat >&2 <<EOF
error: no virtualenv found (looked for $VENV/bin/python and .venv/bin/python).

Create one with Python 3.10 or newer — the package requires >=3.10:

    /opt/homebrew/bin/python3.12 -m venv venv
    venv/bin/pip install -e ".[dev]"
EOF
        exit 1
    fi
fi

PY="$VENV/bin/python"
PY_VER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"

if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    cat >&2 <<EOF
error: $VENV is Python $PY_VER, but this project requires >= 3.10.

CI runs 3.10, 3.11 and 3.12. A 3.9 interpreter cannot import the code
(the \`X | Y\` type syntax and the match statements), and its pytest run
fails at collection with confusing errors.

Create a 3.10+ environment:

    /opt/homebrew/bin/python3.12 -m venv venv
    venv/bin/pip install -e ".[dev]"
EOF
    exit 1
fi

echo "=============================================="
echo " atrium local CI — $ROOT"
echo " interpreter: $VENV (Python $PY_VER)"
echo "=============================================="
echo

# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------
declare -a JOB_NAMES=()
declare -a JOB_RESULTS=()
declare -a JOB_DETAILS=()
# The verdict of the job that just ran. A global rather than "the last
# element of JOB_RESULTS" because macOS ships bash 3.2, which has no
# negative array indices.
LAST_RESULT=""

run_job() {
    local name="$1"; shift
    local required="$1"; shift

    echo "----------------------------------------------"
    echo "▶ $name"
    echo "----------------------------------------------"
    local output
    output="$("$@" 2>&1)"
    local code=$?
    echo "$output"
    echo

    local verdict="PASS"
    [[ $code -ne 0 ]] && verdict="FAIL"
    # An advisory job never fails the run, but its result is still shown.
    # It gets its own verdict string so the summary does not count it as
    # a blocking failure.
    if [[ $code -ne 0 && "$required" == "advisory" ]]; then
        verdict="ADVISORY-FAIL (does not block the merge)"
    fi

    LAST_RESULT="$verdict"
    JOB_NAMES+=("$name")
    JOB_RESULTS+=("$verdict")
    # Keep the last meaningful line as the one-line detail.
    JOB_DETAILS+=("$(echo "$output" | grep -v '^$' | tail -1 | cut -c1-90)")
    return 0
}

# ---------------------------------------------------------------------------
# 1. Lint
# ---------------------------------------------------------------------------
run_job "Lint" required "$VENV/bin/ruff" check atrium tests scripts

# ---------------------------------------------------------------------------
# 2. Typecheck (advisory, same as CI)
# ---------------------------------------------------------------------------
run_job "Typecheck (advisory)" advisory "$VENV/bin/mypy" atrium

# ---------------------------------------------------------------------------
# 3. Test
# ---------------------------------------------------------------------------
run_job "Test (Python $PY_VER)" required "$VENV/bin/pytest" tests/ -q

if [[ $FAST -eq 1 ]]; then
    echo "▶ Build distributions — skipped (--fast)"
    echo "▶ Install the built artifact — skipped (--fast)"
    echo
    JOB_NAMES+=("Build distributions" "Install the built artifact")
    JOB_RESULTS+=("SKIP" "SKIP")
    JOB_DETAILS+=("--fast" "--fast")
else
    # -----------------------------------------------------------------------
    # 4. Build distributions
    # -----------------------------------------------------------------------
    # CI installs `build` and `twine` in this job rather than in `.[dev]`.
    # Mirror that, so the local run exercises the same tools.
    if ! "$PY" -c 'import build' 2>/dev/null; then
        echo "▶ installing 'build' (CI does this in the build job)"
        "$VENV/bin/pip" install --quiet build
    fi
    if ! "$PY" -c 'import twine' 2>/dev/null; then
        echo "▶ installing 'twine'"
        "$VENV/bin/pip" install --quiet twine
    fi
    echo

    run_job "Build distributions" required "$PY" -m build

    if [[ "$LAST_RESULT" == "PASS" ]]; then
        run_job "Build distributions (twine check)" required \
            "$VENV/bin/twine" check dist/*

        # -------------------------------------------------------------------
        # 5. Install the built artifact — the job that catches a missing
        #    module, a broken MANIFEST.in or a console script that does not
        #    land. It is the one CI job that tests what a user actually gets.
        # -------------------------------------------------------------------
        SMOKE_VENV="/tmp/atrium-ci-smoke"
        rm -rf "$SMOKE_VENV"
        echo "----------------------------------------------"
        echo "▶ Install the built artifact"
        echo "----------------------------------------------"
        "$PY" -m venv "$SMOKE_VENV"
        "$SMOKE_VENV/bin/pip" install --quiet dist/*.whl
        # Run from /tmp so `import atrium` resolves to the INSTALLED package
        # and not to the source tree next door.
        ( cd /tmp && "$SMOKE_VENV/bin/python" "$ROOT/scripts/smoke_test.py" )
        smoke_code=$?
        echo
        JOB_NAMES+=("Install the built artifact")
        if [[ $smoke_code -eq 0 ]]; then
            JOB_RESULTS+=("PASS")
        else
            JOB_RESULTS+=("FAIL")
        fi
        JOB_DETAILS+=("smoke test exit=$smoke_code")
        echo
    else
        echo "▶ Install the built artifact — skipped (the build failed)"
        echo
        JOB_NAMES+=("Install the built artifact")
        JOB_RESULTS+=("SKIP")
        JOB_DETAILS+=("build failed")
    fi
fi

# ---------------------------------------------------------------------------
# 6. Build the container image
# ---------------------------------------------------------------------------
if [[ $WITH_DOCKER -eq 1 ]]; then
    if command -v docker >/dev/null 2>&1; then
        run_job "Build the container image" required \
            docker build -t atrium-rag:local .
    else
        echo "▶ Build the container image — skipped (docker not on PATH)"
        echo
        JOB_NAMES+=("Build the container image")
        JOB_RESULTS+=("SKIP")
        JOB_DETAILS+=("docker not installed")
    fi
else
    echo "▶ Build the container image — skipped (pass --with-docker to run it)"
    echo
    JOB_NAMES+=("Build the container image")
    JOB_RESULTS+=("SKIP")
    JOB_DETAILS+=("use --with-docker")
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
FAILED=0
echo "=============================================="
echo " SUMMARY"
echo "=============================================="
for i in "${!JOB_NAMES[@]}"; do
    printf ' %-32s %s\n' "${JOB_NAMES[$i]}" "${JOB_RESULTS[$i]}"
    if [[ "${JOB_RESULTS[$i]}" == "FAIL" ]]; then
        FAILED=1
    fi
done
echo

if [[ $FAILED -eq 0 ]]; then
    echo "All required checks passed."
else
    echo "One or more required checks FAILED — do not open a PR claiming green."
fi
echo

# A paste-ready block for the pull request.
echo "----------------------------------------------"
echo " Paste this into the PR description:"
echo "----------------------------------------------"
echo '```'
echo "\$ scripts/ci_local.sh   # Python $PY_VER"
for i in "${!JOB_NAMES[@]}"; do
    printf ' %-32s %s\n' "${JOB_NAMES[$i]}" "${JOB_RESULTS[$i]}"
done
echo '```'
echo

exit $FAILED
