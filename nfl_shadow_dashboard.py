"""Readable GitHub mobile dashboard for three independent NFL paper engines."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("data/nfl_shadow")
MARKETS = ("ML", "SPREAD", "TOTAL")
NAMES = {"ML": "MONEYLINE · Ganador", "SPREAD": "SPREAD · Hándicap", "TOTAL": "TOTALES · Over/Under"}

def read(name):
    path = ROOT / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def cell(value):
    return str(value if value is not None else "—").replace("|", "/").replace("\n", " ")

def table(rows, columns):
    if not rows:
        return "_Todavía no hay recomendaciones en este mercado._\n"
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cell(row.get(key)) for _, key in columns) + " |" for row in rows]
    return "\n".join([header, divider] + body) + "\n"

def render():
    picks = read("independent_recommendations.jsonl")
    results = {r["recommendation_id"]: r for r in read("independent_results.jsonl")}
    snapshots = read("independent_snapshots.jsonl")
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    parts = [
        "# NFL · Panel de los tres motores independientes",
        "",
        "> **SOLO MONITOREO — $0 apostados.** Actualizado: " + now,
        "",
        "Consulta los resultados desde el celular. Las probabilidades son estimaciones de los modelos, no garantías.",
        "",
        "## Resumen por motor",
        "",
    ]
    summary = []
    for market in MARKETS:
        group = [p for p in picks if p.get("market") == market]
        settled = [results[p["recommendation_id"]] for p in group if p["recommendation_id"] in results]
        wins = sum(r["result"] == "WIN" for r in settled)
        losses = sum(r["result"] == "LOSS" for r in settled)
        pushes = sum(r["result"] == "PUSH" for r in settled)
        roi = 100 * sum(r["paper_return_per_unit"] for r in settled) / len(settled) if settled else None
        summary.append({
            "motor": market, "picks": len(group), "pending": len(group) - len(settled),
            "wins": wins, "losses": losses, "pushes": pushes,
            "hit": f"{100 * wins / (wins + losses):.1f}%" if wins + losses else "—",
            "roi": f"{roi:+.1f}%" if roi is not None else "—",
        })
    parts.extend([table(summary, [
        ("Motor", "motor"), ("Picks", "picks"), ("Pendientes", "pending"),
        ("Ganadas", "wins"), ("Perdidas", "losses"), ("Anuladas", "pushes"),
        ("Acierto", "hit"), ("ROI simulado", "roi"),
    ]), ""])
    if snapshots:
        latest = snapshots[-1]
        parts.append("**Último escaneo:** " + cell(latest.get("captured_at")) +
                     " · " + str(len(latest.get("recommendations", []))) + " lados cotizados.")
        missing = [x.get("game") for x in latest.get("diagnostics", []) if x.get("status") != "QUOTED"]
        if missing:
            parts.append("**Mercados incompletos:** " + ", ".join(missing))
        parts.append("")
    for market in MARKETS:
        group = sorted((p for p in picks if p.get("market") == market),
                       key=lambda p: p.get("captured_at", ""), reverse=True)
        parts.extend(["---", "", "## " + NAMES[market], ""])
        rows = []
        for p in group[:100]:
            result = results.get(p["recommendation_id"])
            status = {"WIN": "GANADA", "LOSS": "PERDIDA", "PUSH": "ANULADA"}.get(
                result["result"], "PENDIENTE") if result else "PENDIENTE"
            score = (f'{result["away_score"]:g}–{result["home_score"]:g}' if result else "—")
            rows.append({
                "date": p.get("captured_at", "")[:10], "game": p.get("game"),
                "pick": p.get("pick"), "prob": f'{p.get("probability", 0):g}%',
                "odds": p.get("odds"), "edge": f'{p.get("edge_pp", 0):g} pp',
                "status": status, "score": score,
            })
        parts.extend([table(rows, [
            ("Fecha", "date"), ("Partido", "game"), ("Selección", "pick"),
            ("Probabilidad", "prob"), ("Cuota", "odds"), ("Ventaja", "edge"),
            ("Resultado", "status"), ("Marcador visita–local", "score"),
        ]), ""])
    parts.extend([
        "---", "",
        "**Actualización:** domingo a las 8:00 a. m. (Hidalgo). "
        "Los partidos terminados se liquidan durante la siguiente ejecución. "
        "Las cuotas originales se conservan para evaluar cada recomendación.",
        "",
        "**Limitaciones:** el origen en tiempo real de los datos de QB, PBP y "
        "pronósticos meteorológicos todavía no está verificado.",
        "",
        "[Ver registros técnicos](independent_snapshots.jsonl) · "
        "[Ver resultados completos](independent_results.jsonl)",
        "",
    ])
    return "\n".join(parts)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / "PANEL_NFL.md"
    target.write_text(render(), encoding="utf-8")
    print(f"Panel actualizado: {target}")

if __name__ == "__main__":
    main()
