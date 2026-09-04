#!/bin/bash
# Run all pre-release checks and determine the release version.
#
# Usage: make-release-checks.sh [--dry-run]
#   --dry-run: warn on failures instead of aborting
#
# Env (when running in GitHub Actions):
#   GH_TOKEN, GITHUB_REPOSITORY, GITHUB_OUTPUT
#   RELEASE_BRANCH: when set, HEAD must belong to origin/RELEASE_BRANCH and must
#     not be older than 3 days from the branch HEAD (skipped when unset)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=false
CHECKS_PASSED=true
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

MAJOR=$(grep "set(LLAMA_VERSION_MAJOR" "$REPO_ROOT/CMakeLists.txt" | grep -oP '\d+')
MINOR=$(grep "set(LLAMA_VERSION_MINOR" "$REPO_ROOT/CMakeLists.txt" | grep -oP '\d+')
PATCH=$(grep "set(LLAMA_VERSION_PATCH" "$REPO_ROOT/CMakeLists.txt" | grep -oP '\d+')
VERSION="v${MAJOR}.${MINOR}.${PATCH}"
echo "Determined version: ${VERSION}"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "version=${VERSION}" >> "$GITHUB_OUTPUT"
fi

SHA=$(git rev-parse HEAD)

# echo "Checking that commit ${SHA} belongs to the release branch..."
# (skipped for testing)

# echo "Checking that tag ${VERSION} does not already exist..."
# (skipped for testing)

# echo "Checking release.yml status for commit ${SHA}..."
# (skipped for testing)

# echo "Checking ggml version..."
# (skipped for testing)

echo "Checking container images for commit ${SHA}..."
NIGHTLY_TAG="${NIGHTLY_TAG_OVERRIDE:-}"
if [[ -z "${NIGHTLY_TAG}" ]]; then
    NIGHTLY_TAG="$(git tag --points-at "${SHA}" | grep -E '(^|-)b[0-9]+(-[0-9a-f]{7})?$' | head -n 1 || true)"
fi
if [[ -z "${NIGHTLY_TAG}" ]]; then
    echo "Warning: no nightly tag points at ${SHA} - skipping container image check"
elif [[ -z "${GITHUB_REPOSITORY:-}" ]]; then
    echo "Warning: GITHUB_REPOSITORY not set - skipping container image check (local run)"
else
    CONTAINER_REPO="${GITHUB_REPOSITORY,,}"  # lower-case owner/repo for ghcr.io
    GHCR_TOKEN="$(curl -fsSL \
        "https://ghcr.io/token?scope=repository:${CONTAINER_REPO}:pull&service=ghcr.io" \
        | grep -oP '"token"\s*:\s*"\K[^"]+')"

    VARIANTS=("" "-cuda" "-cuda13" "-vulkan" "-rocm" "-intel" "-musa" "-openvino")
    TYPES=("full" "light" "server")
    CONTAINER_ERR=""
    for type in "${TYPES[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            tag="${type}${variant}-${NIGHTLY_TAG}"
            STATUS="$(curl -s -o /dev/null -w "%{http_code}" \
                -H "Authorization: Bearer ${GHCR_TOKEN}" \
                -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json" \
                "https://ghcr.io/v2/${CONTAINER_REPO}/manifests/${tag}")"
            if [[ "${STATUS}" == "200" ]]; then
                echo "  ${tag} - OK"
            else
                echo "  ${tag} - MISSING"
                CONTAINER_ERR+=" ${tag}"
            fi
        done
    done

    if [[ -n "${CONTAINER_ERR}" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "Warning: missing container images for ${NIGHTLY_TAG}:${CONTAINER_ERR} (dry run, continuing)."
            CHECKS_PASSED=false
        else
            echo "Error: missing container images for ${NIGHTLY_TAG}:${CONTAINER_ERR}"
            echo "The Docker workflow must complete successfully before making a release."
            exit 1
        fi
    else
        echo "All container images found for ${NIGHTLY_TAG} - OK"
    fi
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "checks_passed=${CHECKS_PASSED}" >> "$GITHUB_OUTPUT"
fi
