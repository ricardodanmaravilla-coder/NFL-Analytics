import nfl_data_py as nfl

from modules.nfl_calendar_ui import calendario_semana, semanas_regulares


def main():
    sched = nfl.import_schedules([2026])
    weeks = semanas_regulares(sched)
    assert weeks == list(range(1, 19)), weeks

    reg = sched[sched['game_type'].eq('REG')].copy()
    assert len(reg) == 272, len(reg)

    total = 0
    for week in weeks:
        games = calendario_semana(sched, week)
        assert not games.empty, f'Week {week} vacía'
        assert games['Visitante'].notna().all()
        assert games['Local'].notna().all()
        total += len(games)
    assert total == 272, total
    print(f'2026 calendar OK: weeks={weeks[0]}-{weeks[-1]} regular_games={total}')


if __name__ == '__main__':
    main()
