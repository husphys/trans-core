"""Tkinter view/control layer for the unchanged MEPI monitor backend."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from apps.mepi_monitor.domain_guard.guard import PredictionDomain
from apps.mepi_monitor.inference.engine import FrozenMEPIEngine
from apps.mepi_monitor.live.replay import ReplaySource
from apps.mepi_monitor.profiles.models import TransformerProfile, builtin_profiles
from apps.mepi_monitor.ui_tk.controller import (
    MonitorWorker, WorkerMessage, aggregate_screening_rows, domain_display,
    load_model_information, make_custom_profile, repository_demo,
    screen_compatible_csv, test_scope_connection,
)

LIMITATIONS = """• Efficiency and core loss are MEPI predictions, not direct two-channel power measurements.
• LSP is a dimensionless relative thermal-stress proxy, not lifetime or RUL.
• Predicted LSP sigma is not calibrated uncertainty.
• Domain membership does not prove generalization.
• Custom-core predictions are exploratory.
• Frequency screening is descriptive and is not experimentally validated optimization.
• Hard real-time performance is not claimed."""


class PlotPanel(ttk.Frame):
    def __init__(self, parent, title: str, ylabel: str = "") -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(5.3, 2.25), dpi=90, layout="constrained")
        self.axis = self.figure.add_subplot(111)
        self.axis.set_title(title); self.axis.set_ylabel(ylabel); self.axis.grid(alpha=.25)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw_series(self, series: list[tuple[object, object, str]], xlabel: str = "Sample") -> None:
        self.axis.clear(); self.axis.grid(alpha=.25); self.axis.set_xlabel(xlabel)
        for x, y, label in series: self.axis.plot(x, y, linewidth=1.1, label=label)
        if len(series) > 1: self.axis.legend(loc="best", fontsize=8)
        self.canvas.draw_idle()


def _group(parent, title: str) -> ttk.LabelFrame:
    return ttk.LabelFrame(parent, text=title, padding=7)


class CustomProfileDialog(tk.Toplevel):
    defaults = {
        "name": "Custom", "od_mm": "34.58", "id_mm": "20.81", "height_mm": "17.68",
        "primary_turns": "10", "secondary_turns": "10", "ae_mode": "geometric",
        "effective_area_m2": "0.0001217268", "material_label": "", "load_resistance_ohm": "49.6025",
    }

    def __init__(self, parent) -> None:
        super().__init__(parent); self.title("Custom Transformer"); self.transient(parent); self.grab_set()
        self.result: TransformerProfile | None = None; self.variables = {}
        labels = {
            "name":"Profile name", "od_mm":"OD (mm)", "id_mm":"ID (mm)", "height_mm":"Height (mm)",
            "primary_turns":"Np", "secondary_turns":"Ns", "ae_mode":"Ae mode",
            "effective_area_m2":"Manual Ae (m²)", "material_label":"Material label",
            "load_resistance_ohm":"Load resistance (ohm)",
        }
        for row, (key, value) in enumerate(self.defaults.items()):
            ttk.Label(self, text=labels[key]).grid(row=row, column=0, sticky="w", padx=8, pady=3)
            variable = tk.StringVar(value=value); self.variables[key] = variable
            widget = ttk.Combobox(self, textvariable=variable, values=("geometric", "manual"), state="readonly") if key == "ae_mode" else ttk.Entry(self, textvariable=variable)
            widget.grid(row=row, column=1, sticky="ew", padx=8, pady=3)
        buttons = ttk.Frame(self); buttons.grid(row=len(self.defaults), column=0, columnspan=2, pady=8)
        ttk.Button(buttons, text="Save", command=self._save).pack(side="left", padx=4)
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="left", padx=4)

    def _save(self) -> None:
        try: self.result = make_custom_profile({key: value.get() for key, value in self.variables.items()})
        except Exception as error: messagebox.showerror("Custom profile", str(error), parent=self); return
        self.destroy()


class MainWindow:
    def __init__(self, root: tk.Tk, project_root: Path, engine: FrozenMEPIEngine, domain: PredictionDomain) -> None:
        self.root, self.project_root, self.engine, self.domain = root, project_root, engine, domain
        self.replay = ReplaySource(project_root / "apps/mepi_monitor/assets/replay/manifest.json")
        self.worker = MonitorWorker(engine, domain, self.replay)
        self.profiles = builtin_profiles(); self.profile = self.profiles["FE"]
        self.connection = tk.StringVar(value="DISCONNECTED")
        self.status = tk.StringVar(value="Ready — frozen CPU inference")
        root.title("MEPI Monitor"); root.geometry("1280x720"); root.minsize(980, 620)
        root.protocol("WM_DELETE_WINDOW", self._close)
        style = ttk.Style(); style.configure("Title.TLabel", font=("TkDefaultFont", 13, "bold")); style.configure("Value.TLabel", font=("TkDefaultFont", 11, "bold"))
        top = ttk.Frame(root, padding=(8, 5)); top.pack(fill="x")
        ttk.Label(top, text="MEPI Monitor", style="Title.TLabel").pack(side="left")
        ttk.Button(top, text="Model Information", command=self._model_information).pack(side="right", padx=4)
        ttk.Button(top, text="About / Limitations", command=lambda: messagebox.showinfo("Scientific limitations", LIMITATIONS)).pack(side="right")
        self.tabs = ttk.Notebook(root); self.tabs.pack(fill="both", expand=True, padx=8, pady=3)
        self.tabs.add(self._live_tab(), text="Live Monitor"); self.tabs.add(self._offline_tab(), text="Offline Screening")
        ttk.Label(root, textvariable=self.status, anchor="w", relief="sunken").pack(fill="x", side="bottom")
        self.root.after(100, self._poll_worker)

    def _live_tab(self):
        page = ttk.Frame(self.tabs, padding=6)
        pane = ttk.Panedwindow(page, orient="horizontal"); pane.pack(fill="both", expand=True)
        controls = ttk.Frame(pane, width=280); plots = ttk.Frame(pane); values = ttk.Frame(pane, width=300)
        pane.add(controls, weight=0); pane.add(plots, weight=1); pane.add(values, weight=0)

        box = _group(controls, "CONNECTION"); box.pack(fill="x", pady=3)
        self.source = tk.StringVar(value="Simulation / Replay"); self.ip = tk.StringVar(value="192.168.2.149"); self.port = tk.StringVar(value="5024")
        for label, widget in (
            ("Source", ttk.Combobox(box, textvariable=self.source, values=("Simulation / Replay", "Keysight LAN"), state="readonly")),
            ("Scope IP", ttk.Entry(box, textvariable=self.ip)), ("Port", ttk.Entry(box, textvariable=self.port))):
            ttk.Label(box, text=label).pack(anchor="w"); widget.pack(fill="x", pady=(0,4))
        buttons = ttk.Frame(box); buttons.pack(fill="x")
        ttk.Button(buttons, text="Connect", command=self._connect).pack(side="left", expand=True, fill="x")
        ttk.Button(buttons, text="Disconnect", command=self._disconnect).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(box, text="Test Scope Connection", command=self._test_scope).pack(fill="x", pady=3)
        ttk.Label(box, textvariable=self.connection, style="Value.TLabel").pack(anchor="w")

        box = _group(controls, "TRANSFORMER PROFILE"); box.pack(fill="x", pady=3)
        self.profile_choice = tk.StringVar(value="FE")
        combo = ttk.Combobox(box, textvariable=self.profile_choice, values=("FE", "Commercial", "Custom Transformer"), state="readonly")
        combo.pack(fill="x"); combo.bind("<<ComboboxSelected>>", self._profile_changed)
        self.profile_info = tk.StringVar(); ttk.Label(box, textvariable=self.profile_info, justify="left", wraplength=260).pack(anchor="w", pady=4); self._show_profile()

        box = _group(controls, "AMBIENT TEMPERATURE"); box.pack(fill="x", pady=3)
        self.ambient = tk.StringVar(value=""); self.interval = tk.StringVar(value="2.0")
        ttk.Label(box, text="Ambient temperature (°C)").pack(anchor="w"); ttk.Entry(box, textvariable=self.ambient).pack(fill="x")
        ttk.Label(box, text="Source: MANUAL INPUT").pack(anchor="w", pady=3)
        ttk.Label(box, text="Update interval (s)").pack(anchor="w"); ttk.Entry(box, textvariable=self.interval).pack(fill="x")
        ttk.Label(box, text="LIVE MONITORING — not hard real-time", wraplength=250).pack(anchor="w", pady=4)
        ttk.Button(box, text="Start Monitoring", command=self._start).pack(fill="x", pady=2)
        ttk.Button(box, text="Stop Monitoring", command=self._stop).pack(fill="x")

        self.wave_plot = PlotPanel(plots, "Vin(t) and Vout(t)", "V"); self.wave_plot.pack(fill="both", expand=True)
        self.b_plot = PlotPanel(plots, "B(t) — physically derived", "T"); self.b_plot.pack(fill="both", expand=True)

        self.value_vars: dict[str, tk.StringVar] = {}
        for title, fields in (
            ("A. MEASURED", ("Frequency", "Vin RMS", "Vout RMS")),
            ("B. PHYSICALLY DERIVED", ("Vin/Vout phase", "Bpeak", "Brms", "B THD", "maximum dB/dt", "form factor")),
            ("C. MEPI PREDICTED", ("Predicted efficiency (%)", "Predicted core loss (W)", "Predicted LSP", "Predicted LSP sigma")),
        ):
            group = _group(values, title); group.pack(fill="x", pady=3)
            for field in fields:
                variable = tk.StringVar(value="—"); self.value_vars[field] = variable
                row = ttk.Frame(group); row.pack(fill="x"); ttk.Label(row, text=field).pack(side="left"); ttk.Label(row, textvariable=variable, style="Value.TLabel").pack(side="right")
        ttk.Label(values, text="Predicted sigma is not calibrated uncertainty.", wraplength=280).pack(anchor="w")
        self.domain_badge = tk.Label(values, text="DOMAIN NOT EVALUATED", bg="#42505c", fg="white", font=("TkDefaultFont", 11, "bold"), padx=8, pady=8, wraplength=280)
        self.domain_badge.pack(fill="x", pady=5)
        details = _group(values, "Feature domain details"); details.pack(fill="both", expand=True)
        self.domain_tree = ttk.Treeview(details, columns=("value","status"), show="headings", height=8)
        for key, title in (("value","Feature / value"),("status","Status")): self.domain_tree.heading(key,text=title)
        self.domain_tree.pack(fill="both", expand=True)
        self.warning = tk.StringVar(value="Inputs are never clamped or silently filled."); ttk.Label(details, textvariable=self.warning, wraplength=280).pack(anchor="w")
        return page

    def _offline_tab(self):
        page = ttk.Frame(self.tabs, padding=8)
        notice = "Frequency screening is descriptive and restricted to the investigated measurement domain. It does not constitute experimentally validated frequency optimization."
        ttk.Label(page, text=notice, wraplength=1100, foreground="#8a5b00").pack(fill="x")
        bar = ttk.Frame(page); bar.pack(fill="x", pady=5)
        self.offline_source = tk.StringVar(value="MEPI 3.9-V Demo Dataset"); self.offline_core = tk.StringVar(value="FE")
        ttk.Combobox(bar, textvariable=self.offline_source, values=("MEPI 3.9-V Demo Dataset", "Compatible imported CSV"), state="readonly", width=30).pack(side="left")
        ttk.Combobox(bar, textvariable=self.offline_core, values=("FE", "COMMERCIAL"), state="readonly", width=14).pack(side="left", padx=4)
        ttk.Button(bar, text="Load / Screen", command=self._load_offline).pack(side="left")
        ttk.Label(bar, text="SCREENED CANDIDATES — no optimum is declared").pack(side="right")
        self.offline_plot = PlotPanel(page, "Descriptive frequency screening"); self.offline_plot.pack(fill="both", expand=True)
        columns = ("frequency","n","efficiency","loss","lsp","sigma","domain")
        self.offline_tree = ttk.Treeview(page, columns=columns, show="headings", height=10)
        for key in columns: self.offline_tree.heading(key, text=key.replace("_"," ").title())
        self.offline_tree.pack(fill="both", expand=True, pady=4); self._load_offline()
        return page

    def _profile_changed(self, _event=None) -> None:
        value = self.profile_choice.get()
        if value == "Custom Transformer":
            dialog = CustomProfileDialog(self.root); self.root.wait_window(dialog)
            if dialog.result is None: self.profile_choice.set("FE"); self.profile = self.profiles["FE"]
            else: self.profile = dialog.result
        else: self.profile = self.profiles["FE" if value == "FE" else "COMMERCIAL"]
        self._show_profile()

    def _show_profile(self) -> None:
        p = self.profile
        self.profile_info.set(f"{p.prediction_label}\nNp={p.primary_turns}  Ns={p.secondary_turns}  Ae={p.effective_area_m2:.7g} m²\nOD={p.od_mm:g}  ID={p.id_mm:g}  height={p.height_mm:g} mm\nLoad={p.load_resistance_ohm:g} Ω  Material={p.material_label or 'unspecified'}")

    def _ambient_value(self):
        text = self.ambient.get().strip()
        if self.source.get() == "Keysight LAN" and not text: raise ValueError("Ambient temperature is required for Keysight live inference")
        return float(text) if text else None

    def _start(self) -> None:
        try:
            replay_index = 0 if self.profile.name == "FE" else 1
            self.worker.start(source=self.source.get(), profile=self.profile, ambient_temperature_c=self._ambient_value(), interval_s=float(self.interval.get()), ip=self.ip.get(), port=int(self.port.get()), replay_index=replay_index)
            self.status.set("Monitoring worker started"); self.connection.set("CONNECTED" if self.source.get() == "Simulation / Replay" else "CONNECTING")
        except Exception as error: messagebox.showerror("Start monitoring", str(error))

    def _stop(self) -> None:
        self.worker.stop(); self.connection.set("DISCONNECTED"); self.status.set("Monitoring stopped")

    def _connect(self) -> None:
        if self.source.get() == "Simulation / Replay": self.connection.set("CONNECTED"); self.status.set("Replay source ready"); return
        self._test_scope()

    def _disconnect(self) -> None: self._stop()

    def _test_scope(self) -> None:
        self.connection.set("CONNECTING")
        def work():
            try: self.root.after(0, lambda identity=test_scope_connection(self.ip.get(), int(self.port.get())): self._scope_result(identity, None))
            except Exception as error: self.root.after(0, lambda value=error: self._scope_result(None, value))
        threading.Thread(target=work, daemon=True, name="scope-idn-test").start()

    def _scope_result(self, identity, error) -> None:
        if error: self.connection.set("ERROR"); messagebox.showerror("Scope connection", str(error))
        else: self.connection.set("CONNECTED"); self.status.set(f"Scope *IDN?: {identity}")

    def _poll_worker(self) -> None:
        self.worker.drain(self._handle_message); self.root.after(100, self._poll_worker)

    def _handle_message(self, message: WorkerMessage) -> None:
        if message.kind == "result": self._show_result(message.payload)
        elif message.kind == "connection": self.connection.set(str(message.payload["status"])); self.status.set(f"Scope: {message.payload['identity']}")
        elif message.kind == "error": self.connection.set("ERROR"); messagebox.showerror("Acquisition / inference", str(message.payload))
        elif message.kind == "stopped" and not self.worker.running: self.status.set("Monitoring worker stopped")

    def _show_result(self, result) -> None:
        p, q = result.processed, result.prediction; m, d = p.measured, p.derived
        self.wave_plot.draw_series([(p.time_s, p.vin_v, "Vin CH1"), (p.time_s, p.vout_v, "Vout CH2")], "Time (s)")
        self.b_plot.draw_series([(range(len(p.b_waveform_t)), p.b_waveform_t, "B(t)")], "Resampled point")
        values = {
            "Frequency":f"{m['frequency_hz']:.3f} Hz", "Vin RMS":f"{m['vin_rms_v']:.5f} V", "Vout RMS":f"{m['vout_rms_v']:.5f} V",
            "Vin/Vout phase":f"{d['phase_shift_deg']:.4f}°", "Bpeak":f"{d['B_peak_t']:.6f} T", "Brms":f"{d['B_rms']:.6f} T",
            "B THD":f"{d['B_thd_percent']:.4f}%", "maximum dB/dt":f"{d['dBdt_max']:.4f} T/s", "form factor":f"{d['form_factor']:.6f}",
            "Predicted efficiency (%)":f"{q.efficiency_percent:.4f}", "Predicted core loss (W)":f"{q.core_loss_w:.6f}",
            "Predicted LSP":f"{q.lsp:.6f}", "Predicted LSP sigma":f"{q.lsp_sigma:.6f}",
        }
        for key, value in values.items(): self.value_vars[key].set(value)
        label, color = domain_display(result.domain.status); core = result.domain.core_status
        self.domain_badge.configure(text=f"{label}\n{core}", bg=color)
        for item in self.domain_tree.get_children(): self.domain_tree.delete(item)
        for name in self.domain.payload["feature_order"]: self.domain_tree.insert("", "end", values=(f"{name}: {p.features[name]:.7g}", result.domain.feature_status[name]))
        self.warning.set("; ".join(result.domain.warnings) or "Inside central training range. Domain membership does not prove generalization.")
        if not self.ambient.get(): self.ambient.set("recorded replay value")
        self.status.set("Inference complete — ambient source: manual/recorded evidence — training performed: NO")

    def _load_offline(self) -> None:
        try:
            if self.offline_source.get() == "MEPI 3.9-V Demo Dataset":
                core = self.offline_core.get(); rows = repository_demo(self.project_root, core); summary = aggregate_screening_rows(rows, self.domain, known_core=True)
            else:
                path = filedialog.askopenfilename(title="Compatible MEPI CSV", filetypes=(("CSV","*.csv"),("All files","*")))
                if not path: return
                rows = screen_compatible_csv(path, self.engine, self.domain); summary = aggregate_screening_rows(rows, self.domain, known_core=False)
            for item in self.offline_tree.get_children(): self.offline_tree.delete(item)
            for row in summary:
                status, _ = domain_display(row["domain_status"])
                self.offline_tree.insert("", "end", values=(f"{row['frequency_hz']:.6g}", row["n"], f"{row['efficiency'][0]:.6g}", f"{row['core_loss'][0]:.6g}", f"{row['lsp'][0]:.6g}", f"{row['lsp_sigma']:.6g}", status))
            x = [row["frequency_hz"] for row in summary]
            self.offline_plot.draw_series([(x,[row["efficiency"][0] for row in summary],"Predicted efficiency"),(x,[row["core_loss"][0] for row in summary],"Predicted core loss"),(x,[row["lsp"][0] for row in summary],"Predicted LSP")], "Frequency (Hz)")
            self.status.set(f"Loaded {len(rows)} screening records; candidates are descriptive only")
        except Exception as error: messagebox.showerror("Offline screening", str(error))

    def _model_information(self) -> None:
        try: payload = load_model_information(self.project_root)
        except Exception as error: messagebox.showerror("Model information", str(error)); return
        lines = ["MEPI v1.5 — 8-layer xLSTM", f"Final test N = {payload['test_rows']}"]
        for key, label in (("efficiency","Efficiency (pp)"),("core_loss","Core loss (W)"),("lsp","LSP")):
            metric = payload["metrics"][key]; text=f"{label}: MAE={metric['mae']:.5g}, RMSE={metric['rmse']:.5g}, R²={metric['r2']:.5g}, Err95={metric['err95_percent']:.5g}%"
            if "mape_percent" in metric: text += f", MAPE={metric['mape_percent']:.5g}%"
            lines.append(text)
        lines.append("\nErr95 is not a confidence interval. Values are loaded from the packaged post-hoc JSON.")
        messagebox.showinfo("Model Information", "\n".join(lines))

    def _close(self) -> None:
        self.worker.stop(); self.root.destroy()


def launch(project_root: Path, engine: FrozenMEPIEngine, domain: PredictionDomain) -> None:
    root = tk.Tk(); MainWindow(root, project_root, engine, domain); root.mainloop()
