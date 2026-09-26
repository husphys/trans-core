"""Research-instrument user interface.

PySide6 is imported only here so the inference engine and headless replay remain
usable on systems that have not installed the GUI extra.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from apps.mepi_monitor.acquisition.keysight_lan import KeysightLanScope
from apps.mepi_monitor.domain_guard.guard import PredictionDomain
from apps.mepi_monitor.inference.engine import FrozenMEPIEngine
from apps.mepi_monitor.live.pipeline import run_capture
from apps.mepi_monitor.live.replay import ReplaySource
from apps.mepi_monitor.offline.loader import load_repository_screening
from apps.mepi_monitor.profiles.models import (
    TransformerProfile, builtin_profiles, geometric_effective_area_m2,
)


class LinePlot(QWidget):
    def __init__(self, title: str, color: str = "#35c2d6") -> None:
        super().__init__()
        self.title, self.color = title, QColor(color)
        self.series: list[tuple[list[float], list[float]]] = []
        self.setMinimumHeight(170)

    def set_series(self, *series: tuple[list[float], list[float]]) -> None:
        self.series = list(series)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101923"))
        painter.setPen(QColor("#d6e4ee"))
        painter.drawText(12, 22, self.title)
        box = QRectF(42, 32, max(10, self.width() - 58), max(10, self.height() - 52))
        painter.setPen(QPen(QColor("#314456"), 1))
        painter.drawRect(box)
        points = [(x, y) for xs, ys in self.series for x, y in zip(xs, ys)]
        if not points:
            painter.drawText(box, Qt.AlignCenter, "No data")
            return
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        if xmax == xmin: xmax += 1.0
        if ymax == ymin: ymax += 1.0
        colors = [self.color, QColor("#f5a742")]
        for index, (sx, sy) in enumerate(self.series):
            poly = []
            for x, y in zip(sx, sy):
                px = box.left() + (x - xmin) / (xmax - xmin) * box.width()
                py = box.bottom() - (y - ymin) / (ymax - ymin) * box.height()
                poly.append(QPointF(px, py))
            painter.setPen(QPen(colors[index % len(colors)], 1.6))
            for a, b in zip(poly, poly[1:]): painter.drawLine(a, b)


class CustomProfileDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Custom Transformer")
        form = QFormLayout(self)
        self.name = QLineEdit("Custom")
        self.material = QLineEdit()
        self.od, self.id_, self.height = (QDoubleSpinBox() for _ in range(3))
        for field, value in zip((self.od, self.id_, self.height), (34.58, 20.81, 17.68)):
            field.setRange(0.001, 10000.0); field.setDecimals(6); field.setValue(value)
        self.np, self.ns = QSpinBox(), QSpinBox()
        for field in (self.np, self.ns): field.setRange(1, 100000); field.setValue(10)
        self.ae_mode = QComboBox(); self.ae_mode.addItems(["Geometric estimate", "Manual"])
        self.ae = QDoubleSpinBox(); self.ae.setDecimals(10); self.ae.setRange(1e-12, 10.0); self.ae.setValue(1.217268e-4)
        self.load = QDoubleSpinBox(); self.load.setRange(1e-6, 1e9); self.load.setDecimals(6); self.load.setValue(49.6025)
        for label, field in (("Profile name", self.name), ("OD (mm)", self.od), ("ID (mm)", self.id_), ("Height (mm)", self.height), ("Np", self.np), ("Ns", self.ns), ("Ae mode", self.ae_mode), ("Effective area Ae (m²)", self.ae), ("Material label (optional)", self.material), ("Load resistance (ohm)", self.load)):
            form.addRow(label, field)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); form.addRow(buttons)

    def profile(self) -> TransformerProfile:
        mode = "geometric" if self.ae_mode.currentIndex() == 0 else "manual"
        area = geometric_effective_area_m2(self.od.value(), self.id_.value(), self.height.value()) if mode == "geometric" else self.ae.value()
        return TransformerProfile(self.name.text(), self.od.value(), self.id_.value(), self.height.value(), self.np.value(), self.ns.value(), area, self.material.text() or None, self.load.value(), False, mode)


def _value_label() -> QLabel:
    label = QLabel("—")
    label.setStyleSheet("font-size: 17px; font-weight: 600; color: #e7f4fa")
    return label


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path, engine: FrozenMEPIEngine, domain: PredictionDomain) -> None:
        super().__init__()
        self.root, self.engine, self.domain = project_root, engine, domain
        self.profiles = builtin_profiles(); self.profile = self.profiles["FE"]
        self.scope: KeysightLanScope | None = None
        self.replay = ReplaySource(self.root / "apps/mepi_monitor/assets/replay/manifest.json")
        self.timer = QTimer(self); self.timer.timeout.connect(self.acquire_once)
        self.setWindowTitle("MEPI Monitor — Frozen v1.5")
        self.resize(1500, 900)
        self.setStyleSheet("QMainWindow,QWidget{background:#0b1219;color:#d6e4ee;font-size:12px} QGroupBox{border:1px solid #294052;border-radius:5px;margin-top:9px;padding-top:10px;font-weight:600} QGroupBox::title{subcontrol-origin:margin;left:8px;color:#7dd8e6} QPushButton{background:#17697a;border:0;border-radius:4px;padding:7px 10px} QPushButton:hover{background:#21849a} QLineEdit,QComboBox,QDoubleSpinBox,QSpinBox{background:#14212c;border:1px solid #355064;padding:4px} QTableWidget{background:#101923;gridline-color:#314456}")
        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._live_tab(), "Live Monitor")
        self.tabs.addTab(self._offline_tab(), "Offline Screening")
        self.statusBar().showMessage(f"Frozen checkpoint {engine.checkpoint_sha256[:12]}… | logging ready")

    def _live_tab(self) -> QWidget:
        page = QWidget(); layout = QGridLayout(page); layout.setColumnStretch(1, 2)
        left = QVBoxLayout(); center = QVBoxLayout(); right = QVBoxLayout()
        profile_box = QGroupBox("Transformer Profile"); pf = QVBoxLayout(profile_box)
        self.profile_combo = QComboBox(); self.profile_combo.addItems(["FE", "Commercial", "Custom Transformer..."])
        self.profile_combo.currentIndexChanged.connect(self._profile_changed); pf.addWidget(self.profile_combo)
        self.profile_note = QLabel(self.profile.prediction_label); self.profile_note.setWordWrap(True); pf.addWidget(self.profile_note)
        left.addWidget(profile_box)
        source_box = QGroupBox("Scope connection / source"); sf = QFormLayout(source_box)
        self.source_combo = QComboBox(); self.source_combo.addItems(["Simulation / Replay", "Keysight LAN"])
        self.ip = QLineEdit("192.168.1.62"); self.connect_button = QPushButton("Connect"); self.connect_button.clicked.connect(self._connect)
        sf.addRow("Source", self.source_combo); sf.addRow("Scope IP", self.ip); sf.addRow(self.connect_button); left.addWidget(source_box)
        input_box = QGroupBox("Ambient temperature and acquisition"); af = QFormLayout(input_box)
        self.ambient = QLineEdit(); self.ambient.setPlaceholderText("Required manual value")
        self.interval = QDoubleSpinBox(); self.interval.setRange(0.25, 3600); self.interval.setValue(2.0); self.interval.setSuffix(" s")
        acquire = QPushButton("Acquire once"); acquire.clicked.connect(self.acquire_once)
        self.monitor = QPushButton("Start monitor"); self.monitor.setCheckable(True); self.monitor.toggled.connect(self._toggle_monitor)
        af.addRow("Ambient temperature", self.ambient); af.addRow("Source", QLabel("Manual input")); af.addRow("Update interval", self.interval); af.addRow(acquire); af.addRow(self.monitor); left.addWidget(input_box)
        self.domain_badge = QLabel("DOMAIN NOT EVALUATED"); self.domain_badge.setAlignment(Qt.AlignCenter); self.domain_badge.setStyleSheet("background:#42505c;padding:12px;font-weight:700")
        left.addWidget(self.domain_badge); left.addStretch()

        self.vin_plot, self.vout_plot, self.b_plot = LinePlot("Vin waveform (MEASURED)"), LinePlot("Vout waveform (MEASURED)", "#f5a742"), LinePlot("B waveform (PHYSICALLY DERIVED)", "#8ee58e")
        for plot in (self.vin_plot, self.vout_plot, self.b_plot): center.addWidget(plot)

        measured = QGroupBox("MEASURED"); mf = QFormLayout(measured)
        self.measured_labels = {key: _value_label() for key in ("frequency", "vin", "vout")}
        for label, key in (("Frequency", "frequency"), ("Vin RMS", "vin"), ("Vout RMS", "vout")): mf.addRow(label, self.measured_labels[key])
        derived = QGroupBox("PHYSICALLY DERIVED"); df = QFormLayout(derived)
        self.derived_labels = {key: _value_label() for key in ("phase", "bpeak", "brms", "thd", "dbdt", "form")}
        for label, key in (("Vin−Vout phase", "phase"), ("Bpeak", "bpeak"), ("Brms", "brms"), ("B THD", "thd"), ("max |dB/dt|", "dbdt"), ("Form factor", "form")): df.addRow(label, self.derived_labels[key])
        predicted = QGroupBox("MEPI PREDICTED"); prf = QFormLayout(predicted)
        self.predicted_labels = {key: _value_label() for key in ("eff", "loss", "lsp", "sigma")}
        for label, key in (("Efficiency (%)", "eff"), ("Core loss (W)", "loss"), ("LSP", "lsp"), ("LSP predicted sigma", "sigma")): prf.addRow(label, self.predicted_labels[key])
        self.prediction_warning = QLabel("Predicted sigma is not calibrated uncertainty."); self.prediction_warning.setWordWrap(True); prf.addRow(self.prediction_warning)
        for box in (measured, derived, predicted): right.addWidget(box)
        right.addStretch()
        layout.addLayout(left, 0, 0); layout.addLayout(center, 0, 1); layout.addLayout(right, 0, 2)
        self.feature_table = QTableWidget(9, 3); self.feature_table.setHorizontalHeaderLabels(["Feature", "Value", "Domain"])
        for row, name in enumerate(self.domain.payload["feature_order"]): self.feature_table.setItem(row, 0, QTableWidgetItem(name))
        bottom = QGroupBox("Domain Guard · Model / checkpoint · Warnings"); bl = QVBoxLayout(bottom); bl.addWidget(self.feature_table)
        self.warning_label = QLabel("No inputs are clamped or modified."); self.warning_label.setWordWrap(True); bl.addWidget(self.warning_label)
        layout.addWidget(bottom, 1, 0, 1, 3)
        return page

    def _offline_tab(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        notice = QLabel("Frequency screening is descriptive and restricted to the investigated measurement domain. It does not constitute experimentally validated frequency optimization.")
        notice.setWordWrap(True); notice.setStyleSheet("background:#3b2d12;color:#ffd98a;padding:10px;font-weight:600"); layout.addWidget(notice)
        controls = QHBoxLayout(); self.offline_core = QComboBox(); self.offline_core.addItems(["FE", "COMMERCIAL"]); self.offline_core.currentTextChanged.connect(self._refresh_offline)
        refresh = QPushButton("Load repository 3.9-V demonstration"); refresh.clicked.connect(self._refresh_offline)
        performance = QPushButton("Model Performance / Test Metrics"); performance.clicked.connect(self._performance_dialog)
        controls.addWidget(self.offline_core); controls.addWidget(refresh); controls.addWidget(performance); controls.addStretch(); layout.addLayout(controls)
        plots = QHBoxLayout(); self.offline_plots = [LinePlot("Frequency vs predicted efficiency"), LinePlot("Frequency vs predicted core loss", "#f5a742"), LinePlot("Frequency vs predicted LSP", "#8ee58e")]
        for plot in self.offline_plots: plots.addWidget(plot); layout.addLayout(plots)
        self.offline_table = QTableWidget(); self.offline_table.setColumnCount(9); self.offline_table.setHorizontalHeaderLabels(["Frequency (Hz)", "N", "Efficiency mean", "Efficiency SD", "Core loss mean", "Core loss SD", "LSP mean", "LSP SD", "Domain"]); layout.addWidget(self.offline_table)
        footer = QLabel("Screened points are candidate operating frequencies only. Repeat variability is the sample standard deviation across measured repeats; predicted LSP sigma is reported separately in the source rows.")
        footer.setWordWrap(True); layout.addWidget(footer); self._refresh_offline(); return page

    def _profile_changed(self, index: int) -> None:
        if index < 2:
            self.profile = self.profiles["FE" if index == 0 else "COMMERCIAL"]
        else:
            dialog = CustomProfileDialog(self)
            if dialog.exec() == QDialog.Accepted:
                self.profile = dialog.profile(); self.profile.validate()
            else:
                self.profile_combo.setCurrentIndex(0); return
        self.profile_note.setText(self.profile.prediction_label)

    def _connect(self) -> None:
        if self.source_combo.currentIndex() == 0:
            self.statusBar().showMessage("Replay source ready; no hardware connection required")
            return
        try:
            if self.scope and self.scope.connected:
                self.scope.disconnect(); self.connect_button.setText("Connect"); return
            self.scope = KeysightLanScope(self.ip.text().strip()); ident = self.scope.connect()
            self.connect_button.setText("Disconnect"); self.statusBar().showMessage(f"Connected: {ident}")
        except Exception as error:
            QMessageBox.critical(self, "Scope connection", str(error))

    def _toggle_monitor(self, enabled: bool) -> None:
        if enabled:
            self.timer.start(int(self.interval.value() * 1000)); self.monitor.setText("Stop monitor"); self.acquire_once()
        else:
            self.timer.stop(); self.monitor.setText("Start monitor")

    def acquire_once(self) -> None:
        try:
            if self.source_combo.currentIndex() == 0:
                capture, entry = self.replay.load(0 if self.profile.name == "FE" else 1)
                ambient = float(entry["ambient_temperature_c"])
                self.ambient.setText(f"{ambient:g}")
            else:
                if not self.scope or not self.scope.connected: raise RuntimeError("Connect to the Keysight scope first")
                text = self.ambient.text().strip()
                ambient = float(text) if text else None
                capture = self.scope.acquire()
            result = run_capture(capture, ambient_temperature_c=ambient, profile=self.profile, engine=self.engine, domain=self.domain)
            p = result.processed; x = list(range(len(p.time_s)))
            self.vin_plot.set_series((x, p.vin_v.tolist())); self.vout_plot.set_series((x, p.vout_v.tolist())); self.b_plot.set_series((list(range(1024)), p.b_waveform_t.tolist()))
            m, d, q = p.measured, p.derived, result.prediction
            for key, text in {"frequency":f"{m['frequency_hz']:.3f} Hz", "vin":f"{m['vin_rms_v']:.5f} V", "vout":f"{m['vout_rms_v']:.5f} V"}.items(): self.measured_labels[key].setText(text)
            values = {"phase":f"{d['phase_shift_deg']:.4f}°", "bpeak":f"{d['B_peak_t']:.6f} T", "brms":f"{d['B_rms']:.6f} T", "thd":f"{d['B_thd_percent']:.4f}%", "dbdt":f"{d['dBdt_max']:.4f} T/s", "form":f"{d['form_factor']:.6f}"}
            for key, text in values.items(): self.derived_labels[key].setText(text)
            for key, text in {"eff":f"{q.efficiency_percent:.4f}", "loss":f"{q.core_loss_w:.6f}", "lsp":f"{q.lsp:.6f}", "sigma":f"{q.lsp_sigma:.6f}"}.items(): self.predicted_labels[key].setText(text)
            colors = {"GREEN":"#176b43", "YELLOW":"#8a6619", "RED":"#862f39"}; color = colors[result.domain.status.split()[0]]
            self.domain_badge.setText(result.domain.status + "\n" + result.domain.core_status); self.domain_badge.setStyleSheet(f"background:{color};padding:12px;font-weight:700")
            for row, name in enumerate(self.domain.payload["feature_order"]):
                self.feature_table.setItem(row, 1, QTableWidgetItem(f"{p.features[name]:.7g}")); self.feature_table.setItem(row, 2, QTableWidgetItem(result.domain.feature_status[name]))
            self.warning_label.setText("; ".join(result.domain.warnings) or "All features are inside the robust central training range. Inputs were not clamped.")
            self.statusBar().showMessage("Inference complete · ambient source: Manual input · logging ready")
        except Exception as error:
            self.timer.stop(); self.monitor.setChecked(False); QMessageBox.critical(self, "Acquisition / inference", str(error))

    def _refresh_offline(self) -> None:
        rows = [row for row in load_repository_screening(self.root) if row["core_id"] == self.offline_core.currentText()]
        grouped = defaultdict(list)
        for row in rows: grouped[float(row["candidate_frequency_hz"])].append(row)
        summary = []
        import statistics
        for frequency in sorted(grouped):
            values = grouped[frequency]
            def stats(key):
                series = [float(row[key]) for row in values]
                return statistics.mean(series), statistics.stdev(series) if len(series) > 1 else 0.0
            statuses = [
                self.domain.assess(
                    {name: float(item[name]) for name in self.domain.payload["feature_order"]},
                    known_core=True,
                ).status
                for item in values
            ]
            rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}
            domain_status = max(statuses, key=lambda item: rank[item.split()[0]])
            summary.append((frequency, len(values), *stats("predicted_efficiency_percent"), *stats("predicted_P_loss"), *stats("predicted_LSP_raw"), domain_status))
        self.offline_table.setRowCount(len(summary))
        for row_index, row in enumerate(summary):
            for column, value in enumerate(row): self.offline_table.setItem(row_index, column, QTableWidgetItem(f"{value:.6g}" if isinstance(value, float) else str(value)))
        frequencies = [row[0] for row in summary]
        for plot, index in zip(self.offline_plots, (2, 4, 6)): plot.set_series((frequencies, [row[index] for row in summary]))

    def _performance_dialog(self) -> None:
        path = self.root / "reports/MEPI_V1_5_POSTHOC_TEST_METRICS.json"
        if not path.exists(): QMessageBox.information(self, "Model Performance", "Post-hoc metrics have not been generated reproducibly."); return
        payload = json.loads(path.read_text(encoding="utf-8")); lines = [f"Final test N: {payload['test_rows']}"]
        for name in ("efficiency", "core_loss", "lsp"):
            metric = payload["metrics"][name]; lines.append(f"{name}: MAE={metric['mae']:.6g}, RMSE={metric['rmse']:.6g}, R²={metric['r2']:.6g}, Err95={metric['err95_percent']:.6g}%" + (f", MAPE={metric['mape_percent']:.6g}%" if "mape_percent" in metric else ""))
        lines.append("\nErr95 is not a confidence interval. Predicted sigma is not calibrated uncertainty.")
        QMessageBox.information(self, "Model Performance / Test Metrics", "\n".join(lines))
