"""Mobile-first GitHub dashboard: one vertically stacked card per NFL paper pick."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("data/nfl_shadow")
MARKETS = (("ML", "🏆 MONEYLINE · Ganador"), ("SPREAD", "📏 SPREAD · Hándicap"),
           ("TOTAL", "🎯 TOTALES · Over/Under"))

def read(filename):
    path = ROOT / filename
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def kelly(probability, american_odds):
    """Hypothetical full Kelly and capped quarter Kelly, percentages."""
    if probability is None or american_odds is None:
        return 0.0, 0.0
    p = float(probability) / 100.0
    odd = float(american_odds)
    if not 0 <= p <= 1 or -100 < odd < 100:
        return 0.0, 0.0
    b = odd / 100 if odd > 0 else 100 / abs(odd)
    full = max(0.0, (b * p - (1 - p)) / b)
    return round(full * 100, 2), round(min(full / 4, 0.05) * 100, 2)

def render(picks=None, settled=None, snapshots=None):
    picks = read("independent_recommendations.jsonl") if picks is None else picks
    settled = read("independent_results.jsonl") if settled is None else settled
    snapshots = read("independent_snapshots.jsonl") if snapshots is None else snapshots
    results = {x["recommendation_id"]: x for x in settled}
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    lines = [
        "# 🏈 PANEL NFL",
        "",
        "### Tres motores independientes · Solo monitoreo",
        "",
        "> **Dinero apostado: $0** · Kelly exclusivamente hipotético.",
        "",
        "**Última actualización:** " + now,
        "",
        "## 📊 Resumen",
        "",
    ]
    for market, name in MARKETS:
        group = [x for x in picks if x.get("market") == market]
        done = [results[x["recommendation_id"]] for x in group if x["recommendation_id"] in results]
        wins = sum(x["result"] == "WIN" for x in done)
        losses = sum(x["result"] == "LOSS" for x in done)
        pushes = sum(x["result"] == "PUSH" for x in done)
        roi = sum(x["paper_return_per_unit"] for x in done) / len(done) * 100 if done else None
        lines.extend([
            "**" + name + "**",
            "",
            f"**{len(group)} picks** · 🟡 {len(group)-len(done)} pendientes · "
            f"🟢 {wins} ganadas · 🔴 {losses} perdidas · ⚪ {pushes} anuladas",
            "",
            "**ROI simulado:** " + (f"{roi:+.1f}%" if roi is not None else "Sin resultados todavía"),
            "",
        ])
    if snapshots:
        latest = snapshots[-1]
        lines.extend(["**Último escaneo:** " + str(latest.get("captured_at", "—"))[:16].replace("T", " ") + " UTC", ""])
    for market, name in MARKETS:
        group = sorted((x for x in picks if x.get("market") == market),
                       key=lambda x: x.get("captured_at", ""), reverse=True)
        lines.extend(["---", "", "## " + name, ""])
        if not group:
            lines.extend(["Sin recomendaciones todavía.", ""])
        for x in group:
            result = results.get(x["recommendation_id"])
            status = {"WIN": "🟢 GANADA", "LOSS": "🔴 PERDIDA",
                      "PUSH": "⚪ ANULADA"}.get(result["result"], "🟡 PENDIENTE") if result else "🟡 PENDIENTE"
            full, quarter = kelly(x.get("probability"), x.get("odds"))
            lines.extend([
                "### " + str(x.get("game", "Partido")),
                "",
                "**🎯 PICK: " + str(x.get("pick", "—")) + "**",
                "",
                "**" + status + "**",
                "",
                f"**Probabilidad:** {x.get('probability', '—')}%  ",
                f"**Cuota:** {x.get('odds', '—')} · {x.get('book') or 'Casa no indicada'}  ",
                f"**Ventaja estimada:** {x.get('edge_pp', '—')} puntos porcentuales  ",
                f"**EV estimado:** {x.get('ev_pct', '—')}%",
                "",
                f"**Kelly teórico:** {full:.2f}%  ",
                f"**¼ Kelly hipotético (máx. 5%):** {quarter:.2f}%",
                "",
                "**Marcador:** " + (f"{result['away_score']:g}–{result['home_score']:g} (visita–local)" if result else "Por jugar"),
                "",
                "<sub>Capturado: " + str(x.get("captured_at", "—"))[:16].replace("T", " ") + " UTC</sub>",
                "",
                "---",
                "",
            ])
    lines.extend([
        "### ℹ️ Información del monitor",
        "",
        "Se ejecuta los **domingos a las 8:00 a. m., hora de Hidalgo**. "
        "Los resultados finalizados se liquidan en la siguiente ejecución.",
        "",
        "El Kelly se calcula con la cuota y la probabilidad originales. "
        "No es una apuesta real ni una garantía de rentabilidad. "
        "Los datos de QB, PBP y meteorología actual todavía no tienen "
        "procedencia prepartido verificada.",
        "",
    ])
    return "\n".join(lines)

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "PANEL_NFL.md"
    path.write_text(render(), encoding="utf-8")
    print("Panel móvil actualizado:", path)
