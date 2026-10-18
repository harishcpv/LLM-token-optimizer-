"""Streamlit UI for the Agent Token Optimizer audit report card."""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
import sys
from typing import Any

try:
    import streamlit as st
except ImportError:  # Keep the CLI and source checkout usable without Streamlit.
    st = None  # type: ignore[assignment]

try:
    from agent_token_optimizer import Pricing, RuleConfig, SimulationConfig
except ModuleNotFoundError:
    source_root = Path(__file__).resolve().parent / "src"
    if not source_root.is_dir():
        raise
    sys.path.insert(0, str(source_root))
    from agent_token_optimizer import Pricing, RuleConfig, SimulationConfig

from audit_cli import AuditResult, run_audit


DEFAULT_CODE = '''def answer(question, search):
    """Answer a question with grounded context."""
    context = search(question)
    return llm(
        system="You are a precise research assistant.",
        user=f"Use this context to answer:\n{context}\n\n{question}",
    )
'''
DEFAULT_PROMPT = """You are a precise research assistant. Use the available tool when the answer depends on current or private information. Cite the relevant evidence, state uncertainty briefly, and keep the final answer under 120 words."""
DEFAULT_TOOLS = '''[
  {
    "type": "function",
    "function": {
      "name": "search",
      "description": "Search the connected knowledge source for relevant evidence to answer the user's question.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "A focused search query containing the key entities and constraints from the question."
          }
        },
        "required": ["query"]
      }
    }
  }
]'''


def _env_float(name: str, fallback: float) -> float:
    try:
        value = float(os.getenv(name, str(fallback)))
    except ValueError:
        return fallback
    return value if value >= 0 else fallback


def _money(value: float) -> str:
    return f"${value:,.4f}"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --ink: #172033; --muted: #64748b; --blue: #2563eb; --teal: #0f766e; }
        [data-testid="stAppViewContainer"] { background: #f5f7fb; }
        [data-testid="stHeader"] { background: rgba(245, 247, 251, 0.88); }
        .block-container { max-width: 1240px; padding-top: 2.2rem; padding-bottom: 3rem; }
        .hero { padding: 1.5rem 1.7rem 1.35rem; border: 1px solid #dbe4f0; border-radius: 22px;
                background: linear-gradient(135deg, #ffffff 0%, #eef5ff 100%); box-shadow: 0 12px 34px rgba(37, 69, 120, .07); }
        .eyebrow { color: #2563eb; font-size: .74rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
        .hero h1 { color: #172033; font-size: clamp(2rem, 4vw, 3.3rem); line-height: 1; margin: .45rem 0 .7rem; }
        .hero p { color: #52627a; font-size: 1.04rem; max-width: 760px; margin: 0; }
        .section-kicker { color: #64748b; font-size: .72rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; margin: .5rem 0 .25rem; }
        .audit-card { border: 1px solid #dbe4f0; border-radius: 18px; background: #fff; padding: 1rem 1.15rem; box-shadow: 0 8px 24px rgba(37, 69, 120, .05); }
        .audit-card h3 { color: #172033; margin: 0; font-size: 1.05rem; }
        .audit-card p { color: #64748b; margin: .35rem 0 0; font-size: .9rem; }
        .pill { display: inline-block; border-radius: 999px; padding: .2rem .55rem; font-size: .73rem; font-weight: 700; background: #e8f0ff; color: #1d4ed8; }
        .muted-note { color: #64748b; font-size: .82rem; }
        div[data-testid="stMetric"] { background: white; border: 1px solid #dbe4f0; border-radius: 15px; padding: .85rem 1rem; }
        div[data-testid="stMetricLabel"] { color: #64748b; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("## Agent Token Optimizer")
        st.caption("A deterministic pre-flight audit for prompts, tools, and agent context.")
        st.divider()
        st.markdown("### Audit assumptions")
        model = st.text_input(
            "Model label",
            value=os.getenv("AGENT_TOKEN_OPTIMIZER_MODEL", "gpt-4o-mini"),
            help="Display label only. The audit never calls a model provider.",
        )
        language = st.selectbox(
            "Source language",
            options=["auto", "python", "javascript", "typescript", "json", "text"],
            index=1,
            help="Used for the report label; token counting remains language-agnostic.",
        )
        input_price = st.number_input(
            "Input price / 1K tokens",
            min_value=0.0,
            value=_env_float("AGENT_TOKEN_OPTIMIZER_INPUT_PER_1K", 0.15),
            step=0.01,
            format="%.4f",
        )
        output_price = st.number_input(
            "Output price / 1K tokens",
            min_value=0.0,
            value=_env_float("AGENT_TOKEN_OPTIMIZER_OUTPUT_PER_1K", 0.60),
            step=0.01,
            format="%.4f",
        )
        st.markdown("### Conversation simulation")
        turns = st.number_input("Turns", min_value=1, max_value=50, value=3, step=1)
        response_tokens = st.number_input(
            "Expected response tokens",
            min_value=0,
            max_value=8192,
            value=256,
            step=16,
        )
        follow_up_tokens = st.number_input(
            "Follow-up tokens per turn",
            min_value=0,
            max_value=4096,
            value=32,
            step=8,
        )
        with st.expander("Finding thresholds", expanded=False):
            verbose_message_tokens = st.number_input(
                "Long message threshold",
                min_value=1,
                max_value=10000,
                value=200,
                step=10,
            )
            verbose_description_tokens = st.number_input(
                "Long description threshold",
                min_value=1,
                max_value=10000,
                value=80,
                step=10,
            )
            bloated_tool_tokens = st.number_input(
                "Bloated tool threshold",
                min_value=1,
                max_value=10000,
                value=256,
                step=16,
            )
        st.divider()
        st.caption("Token backend: local tiktoken when available; otherwise a stable UTF-8 estimate.")
    return {
        "model": model.strip() or "Unspecified model",
        "language": None if language == "auto" else language,
        "pricing": Pricing(input_per_1k=input_price, output_per_1k=output_price),
        "simulation_config": SimulationConfig(
            turns=turns,
            response_tokens=response_tokens,
            follow_up_tokens=follow_up_tokens,
        ),
        "rule_config": RuleConfig(
            verbose_message_tokens=verbose_message_tokens,
            verbose_description_tokens=verbose_description_tokens,
            bloated_tool_tokens=bloated_tool_tokens,
        ),
    }


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Agent Token Optimizer · pre-flight audit</div>
          <h1>Make every agent token earn its place.</h1>
          <p>Paste the context your agent sees, inspect the token footprint, model multi-turn spend, and leave with concrete compression opportunities.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_input_area(settings: dict[str, Any]) -> None:
    with st.form("audit_form", clear_on_submit=False):
        st.markdown('<div class="section-kicker">01 · Ingest agent context</div>', unsafe_allow_html=True)
        st.caption("Everything stays local. No provider API call is made by this scaffold.")
        code_tab, prompt_tab, tools_tab = st.tabs(["Agent code", "Prompt / messages", "Tool schemas"])
        with code_tab:
            uploaded = st.file_uploader(
                "Upload source (optional)",
                type=["py", "js", "ts", "tsx", "jsx", "go", "java", "rs", "rb", "sh", "sql", "txt"],
                help="UTF-8 source is accepted. An upload takes precedence over the pasted code.",
            )
            pasted_code = st.text_area(
                "Paste the agent's orchestration code",
                value=DEFAULT_CODE,
                height=245,
                label_visibility="collapsed",
            )
            if uploaded is not None:
                st.info(f"Using uploaded file: {uploaded.name}")
        with prompt_tab:
            prompt_input = st.text_area(
                "Prompt or provider-shaped JSON",
                value=DEFAULT_PROMPT,
                height=245,
                help='Plain text, a single message object, or JSON with a "messages" array. Embedded tools are accepted too.',
            )
        with tools_tab:
            tools_input = st.text_area(
                "Tool schemas as JSON",
                value=DEFAULT_TOOLS,
                height=245,
                help='Paste one object, an array, or a wrapper like {"tools": [...]}. OpenAI-style function tools are supported.',
            )
            st.caption("Tool descriptions and parameter descriptions are included in the prompt footprint.")
        submitted = st.form_submit_button("Run token audit", type="primary", width="stretch")

    if not submitted:
        return

    code_source = pasted_code
    code_path = None
    if uploaded is not None:
        code_path = uploaded.name
        try:
            code_source = uploaded.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("The uploaded file is not valid UTF-8. Paste a UTF-8 version or choose another file.")
            return

    try:
        result = run_audit(
            code_source=code_source,
            code_path=code_path,
            language=settings["language"],
            prompt_input=prompt_input,
            tools_input=tools_input,
            pricing=settings["pricing"],
            simulation_config=settings["simulation_config"],
            rule_config=settings["rule_config"],
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        st.error("The audit could not parse this input.")
        st.warning(str(exc))
        return
    st.session_state["audit_result"] = result
    st.session_state["audit_model"] = settings["model"]


def _render_overview(result: AuditResult) -> None:
    savings_ratio = result.estimated_savings_ratio
    st.markdown('<div class="section-kicker">02 · Token audit report card</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="pill">{result.counter_backend} counter</span>&nbsp;&nbsp;'
        f'<span class="muted-note">Model assumption: {escape(st.session_state.get("audit_model", "Unspecified model"))}</span>',
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Prompt tokens", f"{result.prompt_tokens:,}", "messages + tools")
    metric_columns[1].metric("Analyzed footprint", f"{result.code_tokens + result.prompt_tokens:,}", "code + prompt")
    metric_columns[2].metric("One-turn estimate", _money(result.one_turn_cost), "input + response")
    metric_columns[3].metric("Savings opportunity", f"{result.estimated_savings_tokens:,}", f"{savings_ratio:.1%} of prompt")

    st.markdown("### Footprint breakdown")
    breakdown_columns = st.columns(4)
    breakdown_columns[0].metric("Message tokens", f"{result.message_tokens:,}")
    breakdown_columns[1].metric("Tool schema tokens", f"{result.tool_tokens:,}")
    breakdown_columns[2].metric("Code tokens", f"{result.code_tokens:,}")
    breakdown_columns[3].metric("Source lines", f"{result.code.line_count:,}")
    st.caption("Code is reported as an analyzed source footprint. Prompt cost includes messages and tool schemas only.")


def _render_savings(result: AuditResult) -> None:
    st.markdown("### Savings recommendations")
    savings_columns = st.columns([1, 1, 2])
    savings_columns[0].metric("Current prompt", f"{result.prompt_tokens:,} tokens")
    savings_columns[1].metric("After recommendations", f"{result.optimized_prompt_tokens:,} tokens")
    savings_columns[2].metric(
        "Estimated one-turn reduction",
        _money(result.one_turn_cost_savings),
        f"up to {result.estimated_savings_ratio:.1%} fewer prompt tokens",
    )
    if result.estimated_savings_tokens:
        savings_ratio = result.estimated_savings_ratio
        st.progress(savings_ratio, text=f"Estimated reduction: {savings_ratio:.1%}")
    else:
        st.info("No savings were estimated by the built-in rules. The prompt is still fully counted and simulated.")

    if not result.findings:
        st.success("No built-in findings for the current thresholds. Review the assumptions and run a provider-specific tokenizer before production pricing.")
        return
    for finding in result.findings:
        label = f"{finding.severity.value.upper()} · {finding.title}"
        if finding.estimated_token_savings:
            label += f" · −{finding.estimated_token_savings} tokens"
        with st.expander(label, expanded=finding.severity.value in {"warning", "error"}):
            st.write(finding.description)
            st.markdown(f"**Recommendation:** {finding.recommendation}")
            if finding.location:
                st.caption(f"Location: `{finding.location}` · Rule: `{finding.rule_id}`")


def _render_simulation(result: AuditResult) -> None:
    st.markdown("### Multi-turn simulation")
    st.caption("The baseline prompt is sent on every turn; each non-final turn adds the assumed response and follow-up tokens to history.")
    rows = [
        {
            "Turn": turn.turn_number,
            "Input tokens": turn.input_tokens,
            "Output tokens": turn.output_tokens,
            "Total tokens": turn.total_tokens,
            "Estimated cost": _money(turn.cost),
        }
        for turn in result.simulation.turns
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    total_columns = st.columns(3)
    total_columns[0].metric("Simulated input", f"{result.simulation.total_input_tokens:,}")
    total_columns[1].metric("Simulated output", f"{result.simulation.total_output_tokens:,}")
    total_columns[2].metric("Simulated spend", _money(result.simulation.total_cost))


def _render_export(result: AuditResult) -> None:
    st.markdown("### Export and assumptions")
    st.download_button(
        "Download JSON report",
        data=json.dumps(result.to_dict(), indent=2, sort_keys=True),
        file_name="agent-token-audit.json",
        mime="application/json",
        width="stretch",
    )
    with st.expander("What this report means"):
        st.markdown(
            "- **Token counts** use `tiktoken` with `cl100k_base` when installed; otherwise the package uses a deterministic UTF-8 estimate.\n"
            "- **Costs** are projections from the prices and response length in the sidebar; no provider is contacted.\n"
            "- **Savings** are rule estimates. Findings can overlap, so treat the sum as an opportunity ceiling and validate after editing.\n"
            "- **Simulation** models history growth, not model quality, latency, retries, or provider-specific caching."
        )


def _render_empty_state() -> None:
    st.markdown('<div class="section-kicker">02 · Report card</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="audit-card">
          <h3>Your audit workspace is ready.</h3>
          <p>Use the sample context or replace it with your agent's code, prompt, and JSON tool definitions. Run the audit to see the full report card.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### What you will get")
    columns = st.columns(3)
    columns[0].markdown("**Token footprint**  \nMessages, tool schemas, source lines, tokenizer backend, and a transparent breakdown.")
    columns[1].markdown("**Spend simulation**  \nOne-turn cost plus deterministic multi-turn history growth under your assumptions.")
    columns[2].markdown("**Actionable savings**  \nDuplicate and verbosity findings with locations, recommendations, and estimated reductions.")


def main() -> None:
    if st is None:
        print("Streamlit is not installed. Install the app extra with: python -m pip install -e '.[app]'")
        return
    st.set_page_config(
        page_title="Agent Token Optimizer",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    settings = _render_sidebar()
    _render_header()
    _render_input_area(settings)
    result = st.session_state.get("audit_result")
    if isinstance(result, AuditResult):
        _render_overview(result)
        _render_savings(result)
        _render_simulation(result)
        _render_export(result)
    else:
        _render_empty_state()


if __name__ == "__main__":
    main()
