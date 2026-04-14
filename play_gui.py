import sys
import argparse

from PySide6.QtWidgets import QApplication

from gui import GoMainWindow


def main():
    parser = argparse.ArgumentParser(description="Go GUI")
    parser.add_argument(
        "--komi",
        type=float,
        default=None,
        help="Optional komi override. Default is 7.5.",
    )
    args, qt_args = parser.parse_known_args()

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("Go AI Arena")
    window = GoMainWindow(komi_override=args.komi)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
