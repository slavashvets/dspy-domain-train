from __future__ import annotations

from typing import Any, Dict, Optional

from dspy.utils.callback import BaseCallback


class ConsoleProgress(BaseCallback):
    """Короткий прогресс: считает LM-вызовы, печатает usage если доступен.
    Устойчив к формам outputs: dict, list и вложенные структуры.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.total_prompt = 0
        self.total_completion = 0
        self.total = 0

    # LM-level
    def on_lm_start(self, call_id: str, instance: Any, inputs: Dict[str, Any]) -> None:
        self.calls += 1
        print(f"[{self.calls}] …", flush=True)

    def on_lm_end(
        self,
        call_id: str,
        outputs: Any = None,
        exception: Optional[Exception] = None,
    ) -> None:
        # Колбэки не должны падать — только логируем.
        try:
            if exception:
                print(f"[LM {self.calls}] error: {exception}", flush=True)
                return

            tokens = self._extract_usage_totals(outputs)
            if tokens is None:
                # usage недоступен — просто отметим завершение
                print(f"[LM {self.calls}] ✓", flush=True)
                return

            pt, ct, tt = tokens
            self.total_prompt += pt
            self.total_completion += ct
            self.total += tt
            print(
                f"[LM {self.calls}] +{tt} tok "
                f"(Σ={self.total} • in={self.total_prompt} • out={self.total_completion})",
                flush=True,
            )
        except Exception as e:
            print(f"[LM {self.calls}] (usage unavailable: {e})", flush=True)

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _extract_usage_totals(outputs: Any) -> Optional[tuple[int, int, int]]:
        """Достаёт usage из произвольной структуры outputs.
        Возвращает (prompt, completion, total) или None.
        """
        if outputs is None:
            return None

        usage_dicts: list[Dict[str, Any]] = []

        def visit(o: Any) -> None:
            if isinstance(o, dict):
                # Предпочитаем явные контейнеры 'usage'/'token_usage'
                for k in ("usage", "token_usage"):
                    v = o.get(k)
                    if isinstance(v, dict):
                        usage_dicts.append(v)

                # Фолбэк: словарь уже выглядит как usage
                keys = set(o.keys())
                if keys & {
                    "prompt_tokens",
                    "input_tokens",
                    "prompt",
                    "completion_tokens",
                    "output_tokens",
                    "completion",
                    "total_tokens",
                }:
                    usage_dicts.append(o)

                for v in o.values():
                    visit(v)
            elif isinstance(o, (list, tuple)):
                for item in o:
                    visit(item)

        visit(outputs)

        if not usage_dicts:
            return None

        # Суммируем всё найденное, избегая дубликатов по id()
        seen: set[int] = set()
        p_sum = c_sum = t_sum = 0

        def as_int(x: Any) -> int:
            try:
                return int(x)
            except Exception:
                return 0

        for u in usage_dicts:
            uid = id(u)
            if uid in seen:
                continue
            seen.add(uid)

            pt = u.get("prompt_tokens") or u.get("input_tokens") or u.get("prompt") or 0
            ct = (
                u.get("completion_tokens")
                or u.get("output_tokens")
                or u.get("completion")
                or 0
            )
            tt = u.get("total_tokens") or (pt or 0) + (ct or 0)

            p_sum += as_int(pt)
            c_sum += as_int(ct)
            t_sum += as_int(tt)

        return p_sum, c_sum, t_sum


def add_usage(u_total: dict, u: Any) -> None:
    """Суммирует usage от Prediction.get_lm_usage().

    Поддерживаемые формы:
    - {lm_name: {prompt_tokens, completion_tokens, total_tokens}}
    - {'usage': {...}, 'model': '...'} или {'token_usage': {...}, ...}
    - [{'model': '...', 'usage': {...}}, ...] (списки)
    - «Плоский» usage-словарь.
    """
    if not u:
        return

    def _acc(name: str, stats: dict) -> None:
        bucket = u_total.setdefault(name, {"input": 0, "output": 0, "total": 0})
        ip = stats.get("prompt_tokens") or stats.get("input_tokens") or stats.get("prompt") or 0
        op = stats.get("completion_tokens") or stats.get("output_tokens") or stats.get("completion") or 0
        tt = stats.get("total_tokens") or (ip or 0) + (op or 0)
        try:
            bucket["input"] += int(ip or 0)
            bucket["output"] += int(op or 0)
            bucket["total"] += int(tt or 0)
        except Exception:
            pass

    # Список записей
    if isinstance(u, list):
        for item in u:
            add_usage(u_total, item)
        return

    # Явная мапа {lm_name: stats}
    if isinstance(u, dict):
        # Случай 1: {lm: {usage...}}
        if all(isinstance(v, dict) for v in u.values()) and any(
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
        ) and not ("usage" in u or "token_usage" in u):
            for lm_name, stats in u.items():
                if isinstance(stats, dict):
                    _acc(str(lm_name), stats)
            return

        # Случай 2: {'usage': {...}, 'model': '...'}
        stats = None
        if isinstance(u.get("usage"), dict):
            stats = u["usage"]
        elif isinstance(u.get("token_usage"), dict):
            stats = u["token_usage"]
        if stats is not None:
            name = str(u.get("model") or u.get("lm") or u.get("name") or "lm")
            _acc(name, stats)
            return

        # Случай 3: плоский usage-словарь
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

    # Иначе — не распознали форму, пропускаем.


def format_usage(u_total: dict) -> str:
    if not u_total:
        return "—"
    parts = []
    for lm, s in u_total.items():
        parts.append(f"{lm}: total={s['total']}, in={s['input']}, out={s['output']}")
    return " | ".join(parts)
