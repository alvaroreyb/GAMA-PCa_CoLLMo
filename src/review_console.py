"""One-key terminal interface to human review."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config_loader import PROJECT_ROOT, output_directory
from .human_review import resolve_review
from .structured_review import resolve_structured_review


def read_key():
    if not sys.stdin.isatty():
        return input().strip().lower()[:1]
    import termios
    import tty
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def default_feedback(judge):
    comments = [judge.get("general_observations", "")]
    comments.extend(f"{k}: {v.get('comment', '')}" for k, v in judge.get("scores", {}).items()
                    if v.get("score", 5) < 5 and v.get("comment"))
    return "\n".join(c for c in comments if c.strip())


def show(payload):
    if payload.get('kind') == 'structured':
        print('\n' + '=' * 76)
        print(payload['case_id'], '— ESQUELETO PENDIENTE —', payload['status'])
        print(json.dumps(payload['case'], ensure_ascii=False, indent=2))
        print('\nHALLAZGOS Y VERIFICACIÓN LOCAL:')
        print(json.dumps(payload['verification'], ensure_ascii=False, indent=2))
        print('\n[A] Validar esqueleto como humano y generar narrativa/juez (consume API)'
              '  [P] Mantener pendiente  [Q] Salir', flush=True)
        return {}
    candidate = payload.get("candidate")
    judge = candidate["judge"] if candidate else payload["judge"]
    print("\n" + "=" * 76)
    print(payload["case_id"], "—", payload["status"])
    if candidate:
        print("Versión corregida", candidate["version"], "— supera criterios:", candidate["passes_acceptance"])
        print("\nNOTA ORIGINAL:\n", payload["narrative"])
    print("\nNOTA A REVISAR:\n", candidate["narrative"] if candidate else payload["narrative"])
    if judge:
        print("\nEVALUACIÓN KIMI:")
        for name, item in judge["scores"].items():
            print(f"  {name}: {item['score']}/5 — {item['comment']}")
        print("Media:", round(sum(v["score"] for v in judge["scores"].values()) / 10, 3))
        for name in ("critical_error", "clinically_acceptable", "requires_revision"):
            print(name, ":", judge[name])
        print("Observaciones:", judge.get("general_observations", ""))
    print("\n[A] Aceptar  [C] Corregir con comentarios de Kimi  [E] Escribir corrección  [P] Pendiente  [Q] Salir", flush=True)
    return judge or {}


def main():
    parser = argparse.ArgumentParser(description="Revisa las notas pendientes mediante una tecla.")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    paths = sorted((output_directory() / "human_review_queue").glob("*_review.json"))
    count = 0
    for path in paths:
        payload = json.loads(path.read_text())
        number = int(payload["case_id"].rsplit("_", 1)[1])
        if not (args.start <= number and (args.end is None or number <= args.end)) or payload.get("status") in ("accepted_by_human", "revised_by_human_request", "ready_for_narrative", "resolved_automatically", "downstream_rejected"):
            continue
        count += 1
        while True:
            judge = show(payload)
            key = read_key()
            if key in ("q", "\x03"):
                return
            if key == "p":
                break
            if payload.get('kind') == 'structured':
                if key != 'a':
                    continue
                try:
                    feedback = input('\nJustificación de la validación humana: ').strip()
                    if not feedback:
                        print('Es necesaria una justificación; se mantiene pendiente.')
                        continue
                    resolve_structured_review(payload['case_id'], feedback)
                    payload = json.loads(path.read_text())
                    break
                except Exception as exc:
                    payload = json.loads(path.read_text())
                    print('No se pudo completar la acción:', exc)
                    continue
            if key not in ("a", "c", "e"):
                continue
            feedback = ""
            if key == "c":
                feedback = default_feedback(judge)
            elif key == "e":
                feedback = input("\nCorrección solicitada: ").strip()
            try:
                print("\nGuardando decisión..." if key == "a" else "\nGLM corrige y Kimi evalúa; espera...", flush=True)
                resolve_review(payload["case_id"], "accept" if key == "a" else "revise", feedback)
                payload = json.loads(path.read_text())
                if key == "a":
                    break
                print("Corrección guardada. Revisa el resultado y pulsa A para aceptarlo.")
            except Exception as exc:
                print("No se pudo completar la acción:", exc)
    print(f"Revisión terminada. Solicitudes mostradas: {count}. Ejecuta de nuevo para cargar nuevas notas.")


if __name__ == "__main__":
    main()
