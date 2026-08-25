from pathlib import Path


def main():
    text = Path('modules/nfl_calendar_ui.py').read_text(encoding='utf-8')
    main_block = text.split('if __name__ == "__main__":', 1)[1]
    assert 'import app_nfl_final' in main_block
    assert 'render_calendario_2026()' not in main_block
    print('calendar helper entrypoint guard OK')


if __name__ == '__main__':
    main()
