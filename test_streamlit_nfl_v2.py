from streamlit.testing.v1 import AppTest


def main():
    at = AppTest.from_file("app_nfl.py", default_timeout=45)
    at.run(timeout=45)
    assert not at.exception, at.exception
    assert len(at.button) >= 1

    # Exercise the production scanner button. External-data failures should be
    # rendered as UI messages, never crash the Streamlit process.
    scan_button = None
    for b in at.button:
        if "Analizar jornada" in b.label:
            scan_button = b
            break
    assert scan_button is not None
    scan_button.click().run(timeout=60)
    assert not at.exception, at.exception
    print("NFL production Streamlit interaction OK")


if __name__ == "__main__":
    main()
