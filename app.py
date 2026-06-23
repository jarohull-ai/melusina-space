"""
Melusina BC Demo — Gradio Space for HuggingFace
JFP Behavioral Constitution Extractor + Mock Agent
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

# ── Import extractor from same directory ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from extractor import Extractor, SignalType, Section, RuleClass

# ── Paths ─────────────────────────────────────────────────────────────────────
CONSTITUTION_PATH = Path(__file__).parent / "constitution.jfp"
CONSTITUTION_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Extractor (no interactive confirm — we handle it in UI) ───────────────────
extractor = Extractor(constitution_path=CONSTITUTION_PATH)

# ── Mock model responses (EN/PL) ──────────────────────────────────────────────
MOCK_RESPONSES = {
    "jfp":       "JFP (Jaro Flash Protocol) is a deterministic behavioral governance protocol for AI agents. "
                 "Current version: v16E.0.0. / "
                 "JFP to deterministyczny protokół zarządzania zachowaniem agentów AI. Wersja v16E.0.0.",
    "melusina":  "MELUSINA is the constitutional AI layer of the JFP ecosystem — "
                 "it extracts behavioral rules from user messages and enforces them via the BC engine. / "
                 "MELUSINA to warstwa konstytucyjna ekosystemu JFP.",
    "lora":      "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique for LLMs. / "
                 "LoRA to technika efektywnego dostrajania modeli językowych.",
    "viki":      "VIKI is the multi-agent AI ecosystem built on JFP, developed by Jarosław Kuchta. / "
                 "VIKI to ekosystem agentów AI oparty na JFP.",
    "hello":     "Hello! I am Melusina — a JFP-governed constitutional AI assistant. How can I help? / "
                 "Witaj! Jestem Melusina — asystent AI oparty na JFP.",
    "default":   "Understood. I am JFP-Core-v1 — a deterministic AI engine. "
                 "I never confabulate: unknown = SIGNAL_UNKNOWN. / "
                 "Rozumiem. Jestem JFP-Core-v1 — deterministyczny agent AI.",
}

def mock_response(message: str) -> str:
    m = message.lower()
    for key, resp in MOCK_RESPONSES.items():
        if key in m:
            return resp
    return MOCK_RESPONSES["default"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_audit_id() -> str:
    ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return f"JFP-{ms}"

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def load_constitution() -> str:
    if not CONSTITUTION_PATH.exists():
        return "*(No rules yet — constitution.jfp is empty)*"
    lines = CONSTITUTION_PATH.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return "*(No rules yet — constitution.jfp is empty)*"
    out = []
    for line in lines:
        try:
            r = json.loads(line)
            cls_icon = {"ALPHA": "🔴", "BETA": "🟡", "GAMMA": "🟢"}.get(r.get("class",""), "⚪")
            src_icon = {"explicit": "📢", "implicit": "↩", "domain": "📚"}.get(r.get("source",""), "❓")
            out.append(
                f"{cls_icon} **{r.get('key','?')}** `{r.get('section','?')}`\n"
                f"   {src_icon} {r.get('value','')}"
            )
        except Exception:
            out.append(line)
    return "\n\n".join(out)

def format_rule_proposal(rule_dict: dict) -> str:
    cls_label = {
        "ALPHA": "🔴 ALPHA — hard constraint / twarda reguła",
        "BETA":  "🟡 BETA — strong preference / silna preferencja",
        "GAMMA": "🟢 GAMMA — domain knowledge / wiedza domenowa",
    }.get(rule_dict.get("class", ""), rule_dict.get("class", ""))
    src_label = {
        "explicit": "📢 EXPLICIT",
        "implicit": "↩ IMPLICIT",
        "domain":   "📚 DOMAIN",
    }.get(rule_dict.get("source", ""), rule_dict.get("source", ""))
    return (
        f"**Section / Sekcja:** `{rule_dict.get('section', '')}`\n"
        f"**Key / Klucz:** `{rule_dict.get('key', '')}`\n"
        f"**Class / Klasa:** {cls_label}\n"
        f"**Source / Źródło:** {src_label}\n"
        f"**Value / Treść:** {rule_dict.get('value', '')}"
    )

# ── State ─────────────────────────────────────────────────────────────────────
# Stored as list of dicts for history panel
_pending_rule: dict | None = None

# ── Core logic ────────────────────────────────────────────────────────────────

def process_message(
    message: str,
    history: list[dict],
    audit_history: list[dict],
):
    """
    Called when user submits a message.
    Returns: (updated_history, audit_history, proposal_md, proposal_json,
               btn_add_visible, btn_reject_visible, constitution_md)
    """
    global _pending_rule

    if not message.strip():
        return history, audit_history, "", "", False, False, load_constitution()

    audit_id  = make_audit_id()
    timestamp = now_iso()

    # Step 1 — detect signal
    signal = extractor.detector.detect(message)

    # Step 2 — mock model response
    agent_reply = mock_response(message)

    # Step 3 — build chat history
    # HF Space (Gradio 5.x) requires messages format: {"role":..., "content":...}
    history = list(history) + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": agent_reply},
    ]

    # Step 4 — if signal detected, generate rule proposal
    proposal_md   = ""
    proposal_json = {}
    show_buttons  = False

    if signal and signal.signal_type != SignalType.NONE:
        next_key = extractor.writer.next_key()
        rule     = extractor.generator.generate(message, signal, next_key)
        rule_d   = rule.to_dict()
        _pending_rule = rule_d

        sig_label = {
            SignalType.EXPLICIT: "📢 EXPLICIT",
            SignalType.IMPLICIT: "↩ IMPLICIT",
            SignalType.DOMAIN:   "📚 DOMAIN",
        }.get(signal.signal_type, str(signal.signal_type))

        proposal_md = (
            f"### Learning signal detected / Wykryto sygnał: {sig_label}\n"
            f"**Match / Dopasowanie:** `{signal.matched_pattern}`\n\n"
            f"---\n"
            f"{format_rule_proposal(rule_d)}"
        )
        proposal_json = json.dumps(rule_d, ensure_ascii=False)  # string for Textbox
        show_buttons  = True

        audit_entry = {
            "audit_id":  audit_id,
            "timestamp": timestamp,
            "signal":    signal.signal_type.value,
            "key":       next_key,
            "status":    "PENDING",
            "message":   message[:60] + ("…" if len(message) > 60 else ""),
        }
    else:
        proposal_json = ""
        _pending_rule = None
        audit_entry = {
            "audit_id":  audit_id,
            "timestamp": timestamp,
            "signal":    "NONE",
            "key":       "—",
            "status":    "NO_RULE",
            "message":   message[:60] + ("…" if len(message) > 60 else ""),
        }

    audit_history = audit_history + [audit_entry]

    return (
        history,
        audit_history,
        proposal_md,
        proposal_json,
        gr.update(visible=show_buttons),
        gr.update(visible=show_buttons),
        load_constitution(),
    )


def add_rule(audit_history: list[dict]):
    """User clicked 'Dodaj regułę'."""
    global _pending_rule
    if _pending_rule is None:
        return audit_history, "*(No pending rule / Brak oczekującej reguły)*", "", "", gr.update(visible=False), gr.update(visible=False), load_constitution()

    extractor.writer.append_dict(_pending_rule)
    key = _pending_rule.get("key", "?")

    # Update last audit entry
    if audit_history:
        last = dict(audit_history[-1])
        last["status"] = "ADDED"
        audit_history  = audit_history[:-1] + [last]

    _pending_rule = None
    return (
        audit_history,
        f"✅ Rule **{key}** added to constitution.jfp / Reguła dodana",
        "",
        "",
        gr.update(visible=False),
        gr.update(visible=False),
        load_constitution(),
    )


def reject_rule(audit_history: list[dict]):
    """User clicked 'Odrzuć'."""
    global _pending_rule
    _pending_rule = None

    if audit_history:
        last = dict(audit_history[-1])
        last["status"] = "REJECTED"
        audit_history  = audit_history[:-1] + [last]

    return (
        audit_history,
        "↩ Rule rejected / Reguła odrzucona.",
        "",
        "",
        gr.update(visible=False),
        gr.update(visible=False),
        load_constitution(),
    )


def format_audit_table(audit_history: list[dict]) -> str:
    if not audit_history:
        return "*(No history yet / Brak historii)*"
    rows = []
    for e in reversed(audit_history[-10:]):  # last 10, newest first
        status_icon = {
            "ADDED":    "✅",
            "REJECTED": "❌",
            "PENDING":  "⏳",
            "NO_RULE":  "💬",
        }.get(e.get("status",""), "❓")
        sig_icon = {
            "explicit": "📢",
            "implicit": "↩",
            "domain":   "📚",
            "NONE":     "💬",
        }.get(e.get("signal",""), "❓")
        rows.append(
            f"{status_icon} `{e.get('audit_id','?')}` · {e.get('timestamp','')} · "
            f"{sig_icon} {e.get('signal','?').upper()} · `{e.get('key','?')}` · "
            f"*{e.get('message','')}*"
        )
    return "\n\n".join(rows)

# ── Patch extractor.writer to support append_dict ────────────────────────────
from extractor import JfpRule, ConstitutionWriter

def _append_dict(self, d: dict) -> None:
    """Append a raw dict as a JSONL line (bypass JfpRule dataclass)."""
    import json as _json
    with open(self.path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(d, ensure_ascii=False) + "\n")

ConstitutionWriter.append_dict = _append_dict

# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
#proposal-box { border: 2px solid #f59e0b; border-radius: 8px; padding: 12px; background: #fffbeb; }
#constitution-box { border: 2px solid #6366f1; border-radius: 8px; padding: 12px; background: #eef2ff; }
#audit-box { border: 2px solid #10b981; border-radius: 8px; padding: 12px; background: #ecfdf5; }
.btn-add { background: #10b981 !important; color: white !important; }
.btn-reject { background: #ef4444 !important; color: white !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Melusina — JFP Constitutional AI Demo") as demo:

    # ── State ──────────────────────────────────────────────────────────────
    audit_state = gr.State([])

    # ── Header ─────────────────────────────────────────────────────────────
    gr.Markdown(
        """
# 🧠 Melusina — JFP Constitutional AI
**Jaro Flash Protocol v16E.0.0** · Behavioral Constitution Extractor

Type a message. If it contains a learning signal (`from now on`, `always`, `never`, `remember that`,
`don't say X we say Y`, `instead of X use Y` — or Polish equivalents),
Melusina will propose adding a rule to your personal constitution.

*Wpisz wiadomość. Jeśli zawiera sygnał uczenia, Melusina zaproponuje regułę do konstytucji.*
        """
    )

    with gr.Row():
        # ── Left column: chat + proposal ───────────────────────────────────
        with gr.Column(scale=3):
            _chatbot_kwargs = {"label": "Melusina", "height": 380, "show_label": True}
            try:
                import inspect as _inspect
                if "type" in _inspect.signature(gr.Chatbot.__init__).parameters:
                    _chatbot_kwargs["type"] = "messages"
            except Exception:
                pass
            chatbot = gr.Chatbot(**_chatbot_kwargs)

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type your message… e.g. 'from now on always respond in English'",
                    label="Your message / Twoja wiadomość",
                    scale=5,
                    lines=1,
                )
                send_btn = gr.Button("Send / Wyślij", variant="primary", scale=1)

            # ── Rule proposal box ─────────────────────────────────────────
            with gr.Group(elem_id="proposal-box"):
                gr.Markdown("#### 💡 Rule Proposal / Propozycja reguły")
                proposal_md   = gr.Markdown("*No learning signal detected. / Brak wykrytego sygnału uczenia.*")
                proposal_json = gr.Textbox(visible=False, label="json_state")

                with gr.Row():
                    btn_add    = gr.Button("✅ Add rule / Dodaj regułę",  elem_classes="btn-add",    visible=False, scale=1)
                    btn_reject = gr.Button("❌ Reject / Odrzuć",          elem_classes="btn-reject",  visible=False, scale=1)

            status_msg = gr.Markdown("")

        # ── Right column: constitution + audit ─────────────────────────────
        with gr.Column(scale=2):
            with gr.Group(elem_id="constitution-box"):
                gr.Markdown("### 📜 constitution.jfp")
                constitution_md = gr.Markdown(load_constitution())
                refresh_btn = gr.Button("🔄 Refresh / Odśwież", size="sm")

            with gr.Group(elem_id="audit-box"):
                gr.Markdown("### 🔍 Audit Log / Historia")
                audit_md = gr.Markdown("*(No history yet / Brak historii)*")

    # ── Examples ───────────────────────────────────────────────────────────
    gr.Markdown("#### 📝 Examples — click to fill / Przykłady — kliknij aby wpisać")
    gr.Examples(
        examples=[
            ["from now on always respond in English"],
            ["never use emoji in responses"],
            ["remember that our project is called VIKI"],
            ["don't say fine-tuning, we say fine-tune"],
            ["instead of LLM use language model"],
            ["what is JFP?"],
            ["tell me about MELUSINA"],
            ["od teraz zawsze odpowiadaj po polsku"],
            ["nigdy nie używaj emoji w odpowiedziach"],
            ["pamiętaj że nasz projekt nazywa się VIKI"],
            ["nie mów 'fine-tuning', mówimy 'dostrajanie'"],
            ["zamiast LLM używaj model językowy"],
        ],
        inputs=msg_input,
    )

    # ── Footer ─────────────────────────────────────────────────────────────
    gr.Markdown(
        "---\n*Melusina BC Demo · JFP v16E.0.0 · "
        "Author / Autor: Jarosław Kuchta · "
        "[GitHub](https://github.com/jarohull-ai) · "
        "[HuggingFace](https://huggingface.co/jarohullowicki)*"
    )

    # ── Event handlers ─────────────────────────────────────────────────────

    def on_submit(message, history, audit_history):
        hist, aud, prop_md, prop_json, btn_a, btn_r, const_md = process_message(
            message, history, audit_history
        )
        audit_display = format_audit_table(aud)
        return hist, aud, prop_md, prop_json, btn_a, btn_r, const_md, audit_display, ""

    def on_add(audit_history):
        aud, status, prop_md, prop_json, btn_a, btn_r, const_md = add_rule(audit_history)
        audit_display = format_audit_table(aud)
        return aud, status, prop_md, prop_json, btn_a, btn_r, const_md, audit_display

    def on_reject(audit_history):
        aud, status, prop_md, prop_json, btn_a, btn_r, const_md = reject_rule(audit_history)
        audit_display = format_audit_table(aud)
        return aud, status, prop_md, prop_json, btn_a, btn_r, const_md, audit_display

    # Submit via button or Enter
    submit_outputs = [
        chatbot, audit_state, proposal_md, proposal_json,
        btn_add, btn_reject, constitution_md, audit_md, msg_input,
    ]
    send_btn.click(on_submit, [msg_input, chatbot, audit_state], submit_outputs)
    msg_input.submit(on_submit, [msg_input, chatbot, audit_state], submit_outputs)

    add_outputs = [
        audit_state, status_msg, proposal_md, proposal_json,
        btn_add, btn_reject, constitution_md, audit_md,
    ]
    btn_add.click(on_add,    [audit_state], add_outputs)
    btn_reject.click(on_reject, [audit_state], add_outputs)

    refresh_btn.click(lambda: load_constitution(), [], [constitution_md])


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=CSS,
    )
