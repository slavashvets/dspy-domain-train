from __future__ import annotations

from typing import Any


def add_usage(u_total: dict, u: Any) -> None:
    """Accumulate usage emitted by Prediction.get_lm_usage().

    Supported shapes:
    - {lm_name: {prompt_tokens, completion_tokens, total_tokens}}
    - {'usage': {...}, 'model': '...'} or {'token_usage': {...}, ...}
    - [{'model': '...', 'usage': {...}}, ...] (lists)
    - Flat usage dictionaries.
    """
    if not u:
        return

    def _acc(name: str, stats: dict) -> None:
        bucket = u_total.setdefault(name, {"input": 0, "output": 0, "total": 0})
        ip = (
            stats.get("prompt_tokens")
            or stats.get("input_tokens")
            or stats.get("prompt")
            or 0
        )
        op = (
            stats.get("completion_tokens")
            or stats.get("output_tokens")
            or stats.get("completion")
            or 0
        )
        tt = stats.get("total_tokens") or (ip or 0) + (op or 0)
        try:
            bucket["input"] += int(ip or 0)
            bucket["output"] += int(op or 0)
            bucket["total"] += int(tt or 0)
        except Exception:
            pass

    if isinstance(u, list):
        for item in u:
            add_usage(u_total, item)
        return

    if isinstance(u, dict):
        if (
            all(isinstance(v, dict) for v in u.values())
            and any(
                {
                    "prompt_tokens",
                    "input_tokens",
                    "prompt",
                    "completion_tokens",
                    "output_tokens",
                    "completion",
                    "total_tokens",
                }
                & set(v.keys())
                for v in u.values()
            )
            and not ("usage" in u or "token_usage" in u)
        ):
            for lm_name, stats in u.items():
                if isinstance(stats, dict):
                    _acc(str(lm_name), stats)
            return

        stats = None
        if isinstance(u.get("usage"), dict):
            stats = u["usage"]
        elif isinstance(u.get("token_usage"), dict):
            stats = u["token_usage"]
        if stats is not None:
            name = str(u.get("model") or u.get("lm") or u.get("name") or "lm")
            _acc(name, stats)
            return

        if {
            "prompt_tokens",
            "input_tokens",
            "prompt",
            "completion_tokens",
            "output_tokens",
            "completion",
            "total_tokens",
        } & set(u.keys()):
            name = str(u.get("model") or u.get("lm") or u.get("name") or "lm")
            _acc(name, u)
            return


def format_usage(u_total: dict) -> str:
    if not u_total:
        return "-"
    parts = []
    for lm, s in u_total.items():
        parts.append(f"{lm}: total={s['total']}, in={s['input']}, out={s['output']}")
    return " | ".join(parts)
