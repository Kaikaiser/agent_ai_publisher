from __future__ import annotations

import sys
from pathlib import Path


def ensure_mcp_vendor_path() -> Path:
    vendor_dir = Path(__file__).resolve().parents[2] / ".vendor"
    vendor_path = str(vendor_dir)
    if vendor_dir.exists() and vendor_path not in sys.path:
        sys.path.append(vendor_path)
    return vendor_dir
