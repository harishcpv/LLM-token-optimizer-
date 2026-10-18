"""Source-checkout wrapper for the packaged Agent Token Optimizer CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_SOURCE = Path(__file__).resolve().parent / "src" / "audit_cli.py"
_SOURCE_ROOT = _SOURCE.parent
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
_spec = importlib.util.spec_from_file_location("_agent_token_optimizer_audit_cli", _SOURCE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"could not load {_SOURCE}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

AuditResult = _module.AuditResult
main = _module.main
parse_tools_input = _module.parse_tools_input
run_audit = _module.run_audit


if __name__ == "__main__":
    raise SystemExit(main())
