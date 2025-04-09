from PyQt5.QtWidgets import QLabel


class SmartLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mouse_position = [0, 0]
        self.setMouseTracking(True)

    def mouseMoveEvent(self, ev):
        self.mouse_position = ev.pos()
