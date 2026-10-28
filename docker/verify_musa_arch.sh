#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Refuse an image whose three MUSA arch selectors disagree.
#
# Three consumers parse the target arch differently, so the image carries the same
# set three ways:
#
#   MTGPU_TARGET          setup.py / vllm_musa's _C build   -> "mp_31" or "mp_31,mp_22"
#   TORCH_MUSA_ARCH_LIST  torch_musa (--offload-arch)       -> "31"   or "31;22"
#   MATE_MUSA_ARCH_LIST   MATE JIT                          -> "3.1"  or "3.1;2.2"
#
# A mismatched set builds the extensions for one arch while the JIT compiles for
# another, and a mubin for the wrong arch does not fault on MUSA: it runs and
# returns wrong numbers.  The check therefore normalises each spelling to a sorted
# set of two-digit codes and compares sets, so separators and the dotted form do
# not matter while a genuinely different arch does.
#
# Run it directly to check a set before building a fat image:
#
#   MTGPU_TARGET=mp_21,mp_22 TORCH_MUSA_ARCH_LIST='21;22' \
#     MATE_MUSA_ARCH_LIST='2.1;2.2' docker/verify_musa_arch.sh

set -euo pipefail

# One normaliser per spelling: commas/semicolons/spaces separate entries, and
# MATE additionally writes the code with a dot ("3.1").
_normalise() {
    local value="$1" dotted="${2:-no}"
    if [ "${dotted}" = "yes" ]; then
        value="$(printf '%s' "${value}" | sed -e 's/\.//g')"
    fi
    printf '%s' "${value}" |
        tr ',;' '\n\n' |
        sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d' -e 's/^mp_//' |
        sort -u |
        paste -sd';'
}

target_codes="$(_normalise "${MTGPU_TARGET:-}")"
torch_codes="$(_normalise "${TORCH_MUSA_ARCH_LIST:-}")"
mate_codes="$(_normalise "${MATE_MUSA_ARCH_LIST:-}" yes)"

if [ -z "${target_codes}" ]; then
    echo "MTGPU_TARGET is empty; it must name at least one arch (e.g. mp_31)" >&2
    exit 1
fi

for code in $(printf '%s' "${target_codes}" | tr ';' ' '); do
    case "${code}" in
        [0-9][0-9]) ;;
        *)
            echo "MTGPU_TARGET must be mp_<two-digit code>, got '${MTGPU_TARGET}'" >&2
            exit 1
            ;;
    esac
done

if [ "${torch_codes}" != "${target_codes}" ]; then
    echo "MTGPU_TARGET='${MTGPU_TARGET}' and TORCH_MUSA_ARCH_LIST='${TORCH_MUSA_ARCH_LIST}'" \
        "name different arches (${target_codes} vs ${torch_codes})" >&2
    exit 1
fi

if [ "${mate_codes}" != "${target_codes}" ]; then
    echo "MTGPU_TARGET='${MTGPU_TARGET}' and MATE_MUSA_ARCH_LIST='${MATE_MUSA_ARCH_LIST}'" \
        "name different arches (${target_codes} vs ${mate_codes})" >&2
    exit 1
fi

echo "MUSA arch selectors agree on ${target_codes}"
