#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<EOF
Usage: $0 user-main [-b build-dir] [-t target-name] [-B backend] [-e] [-h]
 user-main       path to user main source (required)
 -b build-dir    path to build directory (default: ./build/[backend])
 -t target-name  CMake target name to build (default: kocs)
 -B backend      Kokkos backend to request (default: empty -> SERIAL)
 -e              execute built target after successful build
 -h              show this help
EOF
}

BUILD_DIR=""
TARGET_NAME="kocs"
EXECUTE=false
BACKEND=""

if [ $# -lt 1 ]; then
	echo "user-main is required."
	usage
	exit 1
fi
USER_MAIN="$1"
shift

if [[ "$USER_MAIN" == -* ]]; then
	echo "user-main must be provided before flags and cannot start with '-'" >&2
	usage
	exit 1
fi

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
		-t|--target-name)
			if [ -n "${2:-}" ] && [[ "$2" != -* ]]; then
				TARGET_NAME="$2"
				shift 2
			else
				echo "Option $1 requires an argument." >&2
				usage
				exit 1
			fi
			;;
		--target-name=*)
			TARGET_NAME="${1#*=}"
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
		-e|--execute)
			EXECUTE=true
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
	BUILD_DIR="./build/${BACKEND:-SERIAL}"
fi

mkdir -p "$BUILD_DIR"

cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
	-DKOCS_USER_MAIN="$USER_MAIN" \
	-DKOCS_USER_TARGET="$TARGET_NAME" \
	-DKokkos_BACKEND="$BACKEND"

cmake --build "$BUILD_DIR" --target "$TARGET_NAME" -- -j$(nproc || echo 1)

echo "Built target $TARGET_NAME in $BUILD_DIR"

if [ "$EXECUTE" = true ]; then
	EXEC_PATH="$BUILD_DIR/$TARGET_NAME"
	if [ ! -x "$EXEC_PATH" ]; then
		found=$(find "$BUILD_DIR" -maxdepth 4 -type f -name "$TARGET_NAME" -perm /111 2>/dev/null | head -n1 || true)
		if [ -n "$found" ]; then
			EXEC_PATH="$found"
		fi
	fi

	if [ -x "$EXEC_PATH" ]; then
		echo "Executing $EXEC_PATH"
		"$EXEC_PATH"
	else
		echo "Built, but could not find executable '$TARGET_NAME' to run in $BUILD_DIR" >&2
		exit 1
	fi
fi

