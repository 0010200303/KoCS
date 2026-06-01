#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<EOF
Usage: $0 [-b build-dir] [-B backend] [-h help]
 -b build-dir    path to build directory (default: ./build/tests/[backend])
 -B backend      Kokkos backend to request (default: empty -> SERIAL)
 -h help         show this help
EOF
}

BUILD_DIR=""
BACKEND=""

while [ $# -gt 0 ]; do
	case "$1" in
		-b|--build-dir)
			if [ -n "${2:-}" ] && [[ "$2" != -* ]]; then
				BUILD_DIR="$2"
				shift 2
			else
				echo "Option $1 requires an argument." >&2
				usage
				exit 1
			fi
			;;
		--build-dir=*)
			BUILD_DIR="${1#*=}"
			shift
			;;
		-B|--backend)
			if [ -n "${2:-}" ] && [[ "$2" != -* ]]; then
				BACKEND="$2"
				shift 2
			else
				echo "Option $1 requires an argument." >&2
				usage
				exit 1
			fi
			;;
		--backend=*)
			BACKEND="${1#*=}"
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Unknown option: $1" >&2
			usage
			exit 1
			;;
	esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

if [ -z "${BUILD_DIR:-}" ]; then
	BUILD_DIR="$PROJECT_ROOT/build/tests/${BACKEND:-SERIAL}"
fi

mkdir -p "$BUILD_DIR"

echo "Configuring tests in $BUILD_DIR with backend ${BACKEND:-SERIAL}..."
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
	-DKOCS_BUILD_TESTS=ON \
	-DKokkos_BACKEND="$BACKEND"

echo "Building tests..."
cmake --build "$BUILD_DIR" -- -j$(nproc || echo 1)

echo "Running tests..."
cd "$BUILD_DIR"
ctest --output-on-failure
