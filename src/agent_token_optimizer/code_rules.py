"""Static source checks for token-cost anti-patterns."""

from __future__ import annotations

import ast
import re
from typing import Iterable

from .models import CodeArtifact, OptimizationFinding, Severity
from .rules import RuleConfig
from .token_counting import TokenCounter


_PROMPTISH_NAME = re.compile(
    r"(?:prompt|message|system|developer|header|metadata|context|instruction)",
    re.IGNORECASE,
)
_DYNAMIC_CALLS = frozenset(
    {
        "datetime.now",
        "datetime.utcnow",
        "date.today",
        "time.time",
        "time.monotonic",
        "uuid.uuid1",
        "uuid.uuid4",
        "secrets.token_hex",
        "secrets.token_urlsafe",
        "random.random",
        "random.randint",
        "random.choice",
        "uuid4",
    }
)
_DYNAMIC_TEXT = re.compile(
    r"(?:datetime\.(?:now|utcnow)|date\.today|time\.(?:time|monotonic)|"
    r"uuid(?:\.uuid[14]|4)\s*\(|secrets\.token_|random\.)",
    re.IGNORECASE,
)


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    current: ast.AST = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _contains_dynamic_call(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            call_name = _call_name(child)
            if call_name in _DYNAMIC_CALLS or any(
                call_name.endswith(f".{suffix}") for suffix in ("now", "utcnow", "today")
            ):
                return True
    return False


def _target_names(target: ast.AST) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_names(element)


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    return segment or "dynamic value"


def _dynamic_header_finding(
    *,
    source: str,
    node: ast.AST,
    counter: TokenCounter,
    location: str,
) -> OptimizationFinding:
    segment = _source_segment(source, node)
    estimated = max(1, min(counter.count_text(segment), 64))
    return OptimizationFinding(
        rule_id="dynamic-header-prefix",
        severity=Severity.WARNING,
        title="Dynamic prompt/header value may defeat caching",
        description=(
            f"{location} combines a prompt-like value with a per-request clock, ID, "
            "random, or nonce expression. That changes a stable prefix on every call."
        ),
        recommendation=(
            "Keep stable system/developer instructions and tool schemas in a cached "
            "prefix; move request IDs, timestamps, and other dynamic metadata into "
            "a later user/context message or provider metadata field."
        ),
        estimated_token_savings=estimated,
        location=location,
    )


def analyze_code(
    artifact: CodeArtifact,
    *,
    counter: TokenCounter | None = None,
    config: RuleConfig | None = None,
) -> tuple[OptimizationFinding, ...]:
    """Find conservative dynamic-prefix signals in a source artifact."""

    if not isinstance(artifact, CodeArtifact):
        raise TypeError("artifact must be a CodeArtifact")
    counter = counter or TokenCounter()
    config = config or RuleConfig()
    if not config.detect_dynamic_headers or not artifact.source.strip():
        return ()

    findings: list[OptimizationFinding] = []
    if artifact.language in {"python", "py"}:
        try:
            tree = ast.parse(artifact.source)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = node.value
                    if value is None or not _contains_dynamic_call(value):
                        continue
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    names = [name for target in targets for name in _target_names(target)]
                    if any(_PROMPTISH_NAME.search(name) for name in names):
                        line = getattr(node, "lineno", 1)
                        findings.append(
                            _dynamic_header_finding(
                                source=artifact.source,
                                node=node,
                                counter=counter,
                                location=f"line {line}",
                            )
                        )
    else:
        for line_number, line in enumerate(artifact.source.splitlines(), start=1):
            if _PROMPTISH_NAME.search(line) and _DYNAMIC_TEXT.search(line):
                findings.append(
                    _dynamic_header_finding(
                        source=artifact.source,
                        node=ast.Constant(value=line),
                        counter=counter,
                        location=f"line {line_number}",
                    )
                )

    return tuple(sorted(findings, key=lambda finding: (finding.location or "", finding.rule_id)))
