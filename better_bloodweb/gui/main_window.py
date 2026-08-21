"""PyQt6-GUI: Profil-Editor, Kalibrier-Assistent, Run-Panel."""

from __future__ import annotations

import os

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import calibrate, capture
from ..detect import IconDB
from ..profiles import MODES, Profile, list_profiles, load, save
from ..runner import Runner

ASSETS_ICONS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "icons"
)


class RunThread(QThread):
    log_line = pyqtSignal(str)
    finished_run = pyqtSignal()

    def __init__(self, profile: Profile):
        super().__init__()
        self._profile = profile
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        from ..detect import Detector

        runner = Runner(
            self._profile,
            Detector(ASSETS_ICONS),
            log=self.log_line.emit,
            should_stop=lambda: self._stop,
        )
        runner.run()
        self.finished_run.emit()


class CalibrateView(QLabel):
    """Screenshot-Anzeige, Klicks setzen Node-Marker."""

    def __init__(self):
        super().__init__()
        self.nodes: list[tuple[float, float]] = []
        self._pix = None

    def grab_frame(self) -> None:
        frame = capture.capture()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(img)
        self.nodes = []
        self._redraw()

    def mousePressEvent(self, ev) -> None:
        pm = self.pixmap()
        if pm is None:
            return
        pos = ev.position()
        # Position relativ zum angezeigten Pixmap = normalisierte Screenshot-Koordinate
        x = pos.x() / pm.width()
        y = pos.y() / pm.height()
        self.nodes.append((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)))
        self._redraw()

    def _redraw(self) -> None:
        if self._pix is None:
            return
        pm = self._pix.scaled(
            self.width() or 960,
            self.height() or 540,
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        from PyQt6.QtGui import QPainter, QPen, QColor

        pm = pm.copy()
        p = QPainter(pm)
        p.setPen(QPen(QColor(0, 255, 0), 2))
        for i, (nx, ny) in enumerate(self.nodes):
            x, y = int(nx * pm.width()), int(ny * pm.height())
            p.drawEllipse(x - 6, y - 6, 12, 12)
            p.drawText(x + 8, y - 8, str(i))
        p.end()
        self.setPixmap(pm)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("better-bloodweb")
        self.resize(1000, 700)
        self.db = IconDB(ASSETS_ICONS)
        self._run_thread: RunThread | None = None

        tabs = QTabWidget()
        tabs.addTab(self._build_run_tab(), "Run")
        tabs.addTab(self._build_profile_tab(), "Profile")
        tabs.addTab(self._build_calibrate_tab(), "Kalibrierung")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage(
            f"Capture: {capture.backend_name()} | Icons geladen: {len(self.db)}"
        )

    # ---- Run-Tab -----------------------------------------------------------

    def _build_run_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Profil:"))
        self.profile_combo = QComboBox()
        self._refresh_profiles()
        row.addWidget(self.profile_combo)
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        lay.addLayout(row)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log)
        return w

    def _refresh_profiles(self) -> None:
        self.profile_combo.clear()
        self.profile_combo.addItems(list_profiles())

    def _start(self) -> None:
        name = self.profile_combo.currentText()
        if not name:
            QMessageBox.warning(self, "Kein Profil", "Erst im Profile-Tab ein Profil anlegen.")
            return
        if not calibrate.load():
            QMessageBox.warning(self, "Nicht kalibriert", "Bitte zuerst kalibrieren.")
            return
        self._run_thread = RunThread(load(name))
        self._run_thread.log_line.connect(self.log.append)
        self._run_thread.finished_run.connect(self._run_finished)
        self._run_thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log.append("--- Run gestartet (DBD-Bloodweb muss sichtbar sein) ---")

    def _stop(self) -> None:
        if self._run_thread:
            self._run_thread.stop()

    def _run_finished(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ---- Profil-Tab --------------------------------------------------------

    def _build_profile_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        form = QFormLayout()
        self.p_name = QLineEdit()
        self.p_char = QLineEdit()
        self.p_mode = QComboBox()
        self.p_mode.addItems(MODES)
        self.p_bp = QSpinBox()
        self.p_bp.setRange(10_000, 2_000_000)
        self.p_bp.setSingleStep(10_000)
        self.p_bp.setValue(500_000)
        form.addRow("Profilname", self.p_name)
        form.addRow("Charakter", self.p_char)
        form.addRow("Modus", self.p_mode)
        form.addRow("BP-Limit", self.p_bp)
        lay.addLayout(form)

        lay.addWidget(QLabel("Tiers (-999..999, 0 = neutral, nur Abweichungen eintragen):"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self.tier_layout = QVBoxLayout(inner)
        self.tier_spins: dict[str, QSpinBox] = {}
        for item in sorted(self.db.templates):
            row = QHBoxLayout()
            row.addWidget(QLabel(item))
            spin = QSpinBox()
            spin.setRange(-999, 999)
            spin.setValue(0)
            self.tier_spins[item] = spin
            row.addWidget(spin)
            self.tier_layout.addLayout(row)
        if not self.db.templates:
            self.tier_layout.addWidget(QLabel("Keine Icons — erst tools/update-assets.sh laufen lassen."))
        scroll.setWidget(inner)
        lay.addWidget(scroll)

        btns = QHBoxLayout()
        b_save = QPushButton("Speichern")
        b_save.clicked.connect(self._save_profile)
        b_load = QPushButton("Laden")
        b_load.clicked.connect(self._load_profile)
        self.profile_list = QListWidget()
        self.profile_list.addItems(list_profiles())
        btns.addWidget(b_save)
        btns.addWidget(b_load)
        lay.addLayout(btns)
        lay.addWidget(self.profile_list)
        return w

    def _save_profile(self) -> None:
        name = self.p_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Name fehlt", "Profilname angeben.")
            return
        prof = Profile(
            name=name,
            character=self.p_char.text().strip(),
            mode=self.p_mode.currentText(),
            bp_limit=self.p_bp.value(),
            tiers={k: s.value() for k, s in self.tier_spins.items() if s.value() != 0},
        )
        save(prof)
        self.profile_list.clear()
        self.profile_list.addItems(list_profiles())
        self._refresh_profiles()
        self.statusBar().showMessage(f"Profil '{name}' gespeichert.")

    def _load_profile(self) -> None:
        item = self.profile_list.currentItem()
        if not item:
            return
        prof = load(item.text())
        self.p_name.setText(prof.name)
        self.p_char.setText(prof.character)
        self.p_mode.setCurrentText(prof.mode)
        self.p_bp.setValue(prof.bp_limit)
        for k, s in self.tier_spins.items():
            s.setValue(prof.tiers.get(k, 0))

    # ---- Kalibrier-Tab -----------------------------------------------------

    def _build_calibrate_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(
            QLabel(
                "Bloodweb im Spiel öffnen, dann 'Screenshot holen'. "
                "Danach jedes Node-Zentrum anklicken (Reihenfolge egal). Speichern nicht vergessen."
            )
        )
        self.cal_view = CalibrateView()
        self.cal_view.setMinimumSize(640, 360)
        lay.addWidget(self.cal_view)
        btns = QHBoxLayout()
        b_shot = QPushButton("Screenshot holen")
        b_shot.clicked.connect(self.cal_view.grab_frame)
        b_clear = QPushButton("Marker löschen")
        b_clear.clicked.connect(self._cal_clear)
        b_save = QPushButton("Kalibrierung speichern")
        b_save.clicked.connect(self._cal_save)
        btns.addWidget(b_shot)
        btns.addWidget(b_clear)
        btns.addWidget(b_save)
        lay.addLayout(btns)
        existing = calibrate.load()
        self.cal_status = QLabel(f"Gespeicherte Nodes: {len(existing)}")
        lay.addWidget(self.cal_status)
        return w

    def _cal_clear(self) -> None:
        self.cal_view.nodes = []
        self.cal_view._redraw()

    def _cal_save(self) -> None:
        if len(self.cal_view.nodes) < 3:
            QMessageBox.warning(self, "Zu wenig", "Mindestens 3 Nodes markieren.")
            return
        calibrate.save(self.cal_view.nodes)
        self.cal_status.setText(f"Gespeicherte Nodes: {len(self.cal_view.nodes)}")


def run_app() -> None:
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
