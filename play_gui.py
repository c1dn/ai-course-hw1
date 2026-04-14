import sys

from PySide6.QtWidgets import QApplication

from gui import GoMainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Go AI Arena")
    window = GoMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
