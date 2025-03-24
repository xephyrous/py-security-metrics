from PyQt5.QtWidgets import QApplication
from window import Window
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window("C:\\Users\\alexa\\Videos\\guy-walking-across-screen.webm")
    window.setWindowTitle("Py Security Metrics v0.1.a")
    window.setFixedSize(960, 540)
    window.show()

    app.exec()
