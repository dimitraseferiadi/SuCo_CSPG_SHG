#!/usr/bin/env python3
"""
Convert a SuCo index saved by the router (bench_router_paper.py, via the global
faiss.write_index) into the *native* IndexSuCo stream that the per-algorithm
bench_suco_*.py scripts load through IndexSuCo.read_index().

Why this is needed
------------------
faiss.write_index(IndexSuCo) writes:   fourcc("IxSC") + <native SuCo stream>
IndexSuCo.read_index() expects the native stream to start at byte 0 — its header
magic is 0x5375436F ('SuCo'). The leading 4-byte "IxSC" container tag makes the
instance reader fail with "bad magic, not a SuCo index stream". The two on-disk
formats differ ONLY by that 4-byte tag, so the conversion is: drop the fourcc.

The copy is streamed, so even the 10M indexes convert with negligible memory.

Usage:
    convert_suco_index.py <router_src.idx> <native_dst.idx>
"""
import os
import struct
import sys

IXSC = b"IxSC"              # faiss global-container tag for IndexSuCo
SUCO_MAGIC = 0x5375436F     # native IndexSuCo header magic ('SuCo'), little-endian
_CHUNK = 1 << 20


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: convert_suco_index.py <router_src.idx> <native_dst.idx>")
    src, dst = sys.argv[1], sys.argv[2]

    with open(src, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            sys.exit(f"convert: {src!r} is too short to be a SuCo index")
        tag = head[:4]
        magic_after_tag = struct.unpack("<I", head[4:8])[0]
        magic_at_0 = struct.unpack("<I", head[:4])[0]

        if tag == IXSC and magic_after_tag == SUCO_MAGIC:
            # Global container: strip the 4-byte fourcc, keep the native stream.
            tmp = dst + ".tmp"
            with open(tmp, "wb") as g:
                g.write(head[4:])                     # native magic + first bytes
                while True:
                    chunk = f.read(_CHUNK)
                    if not chunk:
                        break
                    g.write(chunk)
            os.replace(tmp, dst)                      # atomic
            print(f"convert: {src} -> {dst}  (stripped IxSC container)")
        elif magic_at_0 == SUCO_MAGIC:
            sys.exit(f"convert: {src!r} is already native SuCo format — "
                     f"symlink or copy it directly, no conversion needed")
        else:
            sys.exit(f"convert: {src!r} unrecognized header "
                     f"(tag={tag!r}, magic={magic_after_tag:#010x}) — "
                     f"not a router-saved IndexSuCo")


if __name__ == "__main__":
    main()

