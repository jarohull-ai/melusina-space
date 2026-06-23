"""
Melusina BC Extractor — JFP Behavioral Constitution Rule Extractor
Pattern-based (no LLM), deterministic signal detection.

Usage:
    python3 extractor.py                    # interactive mode
    python3 extractor.py "od teraz..."      # single message mode
    python3 extractor.py --no-confirm "..."  # skip confirmation (for scripts/tests)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

CONSTITUTION_PATH = Path.home() / ".jfp" / "constitution.jfp"

# ── Enums ─────────────────────────────────────────────────────────────────────

class SignalType(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    DOMAIN   = "domain"
    NONE     = "none"

class Section(str, Enum):
    BEHAVIORAL_RULES = "BEHAVIORAL_RULES"
    DOMAIN_KNOWLEDGE = "DOMAIN_KNOWLEDGE"
    PRIORITIES       = "PRIORITIES"

class RuleClass(str, Enum):
    ALPHA = "ALPHA"   # hard constraint — never override
    BETA  = "BETA"    # strong preference — override only with reason
    GAMMA = "GAMMA"   # soft knowledge — can be updated

# ── Patterns ──────────────────────────────────────────────────────────────────

# Each entry: (compiled_regex, SignalType, Section, RuleClass)
_PATTERNS: list[tuple[re.Pattern, SignalType, Section, RuleClass]] = [

    # ── EXPLICIT / ALPHA — hard behavioral rules ──────────────────────────────
    (re.compile(
        r"\b(zawsze|always|od teraz zawsze|from now on always)\b",
        re.IGNORECASE
    ), SignalType.EXPLICIT, Section.BEHAVIORAL_RULES, RuleClass.ALPHA),

    (re.compile(
        r"\b(nigdy|never|od teraz nigdy|from now on never)\b",
        re.IGNORECASE
    ), SignalType.EXPLICIT, Section.BEHAVIORAL_RULES, RuleClass.ALPHA),

    (re.compile(
        r"\b(od teraz|from now on)\b",
        re.IGNORECASE
    ), SignalType.EXPLICIT, Section.BEHAVIORAL_RULES, RuleClass.ALPHA),

    # ── EXPLICIT / BETA — priorities / memory ─────────────────────────────────
    (re.compile(
        r"\b(pamiętaj że|pamiętaj,?\s*że|remember that|remember:)\b",
        re.IGNORECASE
    ), SignalType.EXPLICIT, Section.PRIORITIES, RuleClass.BETA),

    # ── DOMAIN / GAMMA — terminology corrections ──────────────────────────────
    (re.compile(
        r"\b(to się nazywa|nazywamy to|nasz termin to|we call it|the term is"
        r"|nie mów[\"' ]+\w|don't (say|call it)|it's called)\b",
        re.IGNORECASE
    ), SignalType.DOMAIN, Section.DOMAIN_KNOWLEDGE, RuleClass.GAMMA),

    (re.compile(
        r"\bto nie\b.{1,40}\bto\b",
        re.IGNORECASE
    ), SignalType.DOMAIN, Section.DOMAIN_KNOWLEDGE, RuleClass.GAMMA),

    # ── IMPLICIT / BETA — corrections of previous output ─────────────────────
    (re.compile(
        r"^(nie[,\s]+|wrong[,:\s]+|błąd[,:\s]+|popraw[,:\s]+|actually[,:\s]+|"
        r"correction[,:\s]+|to nieprawda|that'?s (wrong|incorrect))",
        re.IGNORECASE
    ), SignalType.IMPLICIT, Section.BEHAVIORAL_RULES, RuleClass.BETA),

    (re.compile(
        r"\b(nie mów|nie pisz|nie używaj|don't (say|write|use))\b",
        re.IGNORECASE
    ), SignalType.IMPLICIT, Section.BEHAVIORAL_RULES, RuleClass.BETA),
]

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DetectedSignal:
    signal_type: SignalType
    section:     Section
    rule_class:  RuleClass
    matched_pattern: str

@dataclass
class JfpRule:
    section:  str
    key:      str
    value:    str
    cls:      str          # "class" is a Python keyword
    source:   str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "section":   self.section,
            "key":       self.key,
            "value":     self.value,
            "class":     self.cls,
            "source":    self.source,
            "timestamp": self.timestamp,
        }

# ── Signal Detector ───────────────────────────────────────────────────────────

class SignalDetector:
    """
    Scans a message for learning signals using ordered regex patterns.
    Returns the first (highest-priority) match.
    """

    def detect(self, message: str) -> Optional[DetectedSignal]:
        for pattern, sig_type, section, rule_class in _PATTERNS:
            m = pattern.search(message)
            if m:
                return DetectedSignal(
                    signal_type=sig_type,
                    section=section,
                    rule_class=rule_class,
                    matched_pattern=m.group(0),
                )
        return None

# ── Rule Generator ────────────────────────────────────────────────────────────

class RuleGenerator:
    """
    Converts a raw message + detected signal into a JfpRule proposal.
    """

    def generate(self, message: str, signal: DetectedSignal, next_key: str) -> JfpRule:
        value = self._extract_value(message, signal)
        return JfpRule(
            section=signal.section.value,
            key=next_key,
            value=value,
            cls=signal.rule_class.value,
            source=signal.signal_type.value,
        )

    def _extract_value(self, message: str, signal: DetectedSignal) -> str:
        """
        Strip trigger phrases and return the core rule content.
        Normalises whitespace and capitalises first letter.
        """
        triggers = [
            r"\b(od teraz|from now on)\b[,\s]*",
            r"\b(zawsze|always)\b[,\s]*",
            r"\b(nigdy|never)\b[,\s]*",
            r"\b(pamiętaj że|pamiętaj,?\s*że|remember that|remember:)\s*",
            r"\b(to się nazywa|nazywamy to|nasz termin to|we call it|the term is)\s*",
            r"^(nie[,\s]+|wrong[,:\s]+|błąd[,:\s]+|popraw[,:\s]+|actually[,:\s]+)",
        ]
        value = message.strip()
        for t in triggers:
            value = re.sub(t, "", value, flags=re.IGNORECASE).strip()

        # Collapse multiple spaces
        value = re.sub(r"\s{2,}", " ", value)

        # Capitalise
        if value:
            value = value[0].upper() + value[1:]

        return value or message.strip()

# ── Constitution Writer ───────────────────────────────────────────────────────

class ConstitutionWriter:
    """
    Reads and writes rules to ~/.jfp/constitution.jfp (JSONL format).
    """

    def __init__(self, path: Path = CONSTITUTION_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def next_key(self) -> str:
        """Returns the next available RULE_NNN key."""
        existing = self.load_all()
        if not existing:
            return "RULE_001"
        nums = []
        for r in existing:
            m = re.match(r"RULE_(\d+)", r.get("key", ""))
            if m:
                nums.append(int(m.group(1)))
        nxt = max(nums) + 1 if nums else 1
        return f"RULE_{nxt:03d}"

    def load_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        rules = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rules.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return rules

    def append(self, rule: JfpRule) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rule.to_dict(), ensure_ascii=False) + "\n")

# ── Extractor (orchestrator) ──────────────────────────────────────────────────

class Extractor:
    """
    Main pipeline: detect → generate → (confirm) → write.
    """

    def __init__(self, constitution_path: Path = CONSTITUTION_PATH):
        self.detector  = SignalDetector()
        self.generator = RuleGenerator()
        self.writer    = ConstitutionWriter(constitution_path)

    def process(
        self,
        message: str,
        confirm: bool = True,
    ) -> Optional[JfpRule]:
        """
        Process a single message.

        Args:
            message: raw user input
            confirm: if True, ask user interactively before writing

        Returns:
            JfpRule if a rule was written, None otherwise
        """
        signal = self.detector.detect(message)

        if signal is None or signal.signal_type == SignalType.NONE:
            print("ℹ  Brak sygnału uczenia — wiadomość nie zawiera reguły.")
            return None

        next_key = self.writer.next_key()
        rule     = self.generator.generate(message, signal, next_key)

        self._print_proposal(rule, signal)

        if confirm:
            answer = input("\nCzy dodać tę regułę? [T/N]: ").strip().upper()
            if answer not in ("T", "TAK", "Y", "YES"):
                print("↩  Reguła odrzucona.")
                return None

        self.writer.append(rule)
        print(f"✓  Reguła {rule.key} zapisana → {self.writer.path}")
        return rule

    def _print_proposal(self, rule: JfpRule, signal: DetectedSignal) -> None:
        print("\n" + "─" * 56)
        print(f"  Wykryty sygnał : {signal.signal_type.value.upper()}")
        print(f"  Dopasowanie    : '{signal.matched_pattern}'")
        print("─" * 56)
        print("  Propozycja reguły JFP:")
        print(json.dumps(rule.to_dict(), indent=4, ensure_ascii=False))
        print("─" * 56)

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Melusina BC Extractor — JFP rule extractor (pattern-based)"
    )
    parser.add_argument(
        "message",
        nargs="?",
        help="Message to analyse (omit for interactive mode)",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt (auto-accept)",
    )
    parser.add_argument(
        "--constitution",
        default=str(CONSTITUTION_PATH),
        help=f"Path to constitution file (default: {CONSTITUTION_PATH})",
    )
    args = parser.parse_args()

    extractor = Extractor(constitution_path=Path(args.constitution))

    if args.message:
        extractor.process(args.message, confirm=not args.no_confirm)
    else:
        # Interactive loop
        print("Melusina BC Extractor — wpisz wiadomość (Ctrl+C aby wyjść)")
        print("─" * 56)
        try:
            while True:
                try:
                    msg = input("\n> ").strip()
                except EOFError:
                    break
                if not msg:
                    continue
                extractor.process(msg, confirm=not args.no_confirm)
        except KeyboardInterrupt:
            print("\nDo widzenia.")


if __name__ == "__main__":
    main()
