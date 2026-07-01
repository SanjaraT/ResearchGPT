"""
ResearchGPT

The primary application is the Streamlit interface.

Run:

    streamlit run streamlit_app.py

For Docker:

    docker run -p 8501:8501 --env-file .env researchgpt
"""

import subprocess
import sys


def main():

    print("=" * 60)
    print("📚 ResearchGPT")
    print("=" * 60)

    print("\nLaunching Streamlit...\n")

    try:

        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "streamlit_app.py"
            ]
        )

    except KeyboardInterrupt:

        print("\nResearchGPT closed.")


if __name__ == "__main__":
    main()