"""
GGTH Predictor GUI v2.0
Updated for unified_predictor_v8.py
Author: Jason Rusk
"""

import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext


# Updated script name for the new predictor
SCRIPT_NAME = "unified_predictor_v8.py"
VERSION = "2.0"


class GGTHGui(tk.Tk):
    """
    Front-end for unified_predictor_v8.py

    It calls the existing script with command-line modes:
      - train
      - train-multitf
      - tune
      - predict
      - predict-multitf
      - backtest
    """

    def __init__(self):
        super().__init__()
        self.title(f"GGTH Predictor – Control Panel v{VERSION}")
        self.geometry("820x650")
        self.resizable(False, False)

        # State variables
        self.symbol_var = tk.StringVar(value="USDJPY")
        self.action_var = tk.StringVar(value="train-multitf")
        self.force_retrain_var = tk.BooleanVar(value=True)
        self.continuous_var = tk.BooleanVar(value=False)
        self.interval_var = tk.IntVar(value=60)
        
        # Model checkboxes - Added GRU
        self.models_lstm_var = tk.BooleanVar(value=True)
        self.models_gru_var = tk.BooleanVar(value=False)  # NEW: GRU model
        self.models_transformer_var = tk.BooleanVar(value=True)
        self.models_tcn_var = tk.BooleanVar(value=False)
        self.models_lgbm_var = tk.BooleanVar(value=True)
        
        self.use_kalman_var = tk.BooleanVar(value=True)
        self.python_exe_var = tk.StringVar(value=sys.executable)
        self.script_path_var = tk.StringVar(value=self._default_script_path())
        self.mt5_path_var = tk.StringVar(value="")

        # Load MT5 path from config
        self._load_mt5_path()

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        # --- Script / Python config -----------------------------------
        cfg_frame = ttk.LabelFrame(self, text="Python & Script Configuration")
        cfg_frame.place(x=10, y=10, width=800, height=120)

        ttk.Label(cfg_frame, text="Python exe:").place(x=10, y=10)
        python_entry = ttk.Entry(cfg_frame, textvariable=self.python_exe_var, width=72)
        python_entry.place(x=90, y=8)
        ttk.Button(cfg_frame, text="Browse...", command=self._browse_python).place(
            x=710, y=6, width=70
        )

        ttk.Label(cfg_frame, text="Predictor:").place(x=10, y=40)
        script_entry = ttk.Entry(cfg_frame, textvariable=self.script_path_var, width=72)
        script_entry.place(x=90, y=38)
        ttk.Button(cfg_frame, text="Browse...", command=self._browse_script).place(
            x=710, y=36, width=70
        )

        ttk.Label(cfg_frame, text="MT5 Files:").place(x=10, y=70)
        mt5_entry = ttk.Entry(cfg_frame, textvariable=self.mt5_path_var, width=72)
        mt5_entry.place(x=90, y=68)
        ttk.Button(cfg_frame, text="Browse...", command=self._browse_mt5).place(
            x=710, y=66, width=70
        )

        ttk.Label(cfg_frame, text="(MT5 Terminal\\...\\MQL5\\Files directory)",
                  foreground="gray", font=("Segoe UI", 7)).place(x=90, y=90)

        # --- Basic settings -------------------------------------------
        basic_frame = ttk.LabelFrame(self, text="Basic Settings")
        basic_frame.place(x=10, y=135, width=395, height=130)

        ttk.Label(basic_frame, text="Symbol:").place(x=10, y=10)
        ttk.Entry(basic_frame, textvariable=self.symbol_var, width=12).place(x=70, y=8)

        ttk.Label(basic_frame, text="Action:").place(x=10, y=40)
        row_y = 35
        ttk.Radiobutton(
            basic_frame,
            text="Train ALL models (multi-TF) [RECOMMENDED]",
            variable=self.action_var,
            value="train-multitf",
        ).place(x=70, y=row_y)
        row_y += 22
        ttk.Radiobutton(
            basic_frame,
            text="Train main ensemble (single config)",
            variable=self.action_var,
            value="train",
        ).place(x=70, y=row_y)
        row_y += 22
        ttk.Radiobutton(
            basic_frame,
            text="Hyperparameter tuning",
            variable=self.action_var,
            value="tune",
        ).place(x=70, y=row_y)

        # --- Prediction / Backtest modes ------------------------------
        mode_frame = ttk.LabelFrame(self, text="Prediction / Backtest Modes")
        mode_frame.place(x=415, y=135, width=395, height=130)

        ttk.Radiobutton(
            mode_frame,
            text="Predict ONCE for EA (multi-TF JSON)",
            variable=self.action_var,
            value="predict-mtf-once",
        ).place(x=10, y=10)

        ttk.Radiobutton(
            mode_frame,
            text="Predict CONTINUOUSLY (multi-TF JSON)",
            variable=self.action_var,
            value="predict-mtf-cont",
        ).place(x=10, y=35)

        ttk.Radiobutton(
            mode_frame,
            text="Generate backtest predictions",
            variable=self.action_var,
            value="backtest",
        ).place(x=10, y=60)
        
        ttk.Radiobutton(
            mode_frame,
            text="Safe backtest (walk-forward, anti-leakage)",
            variable=self.action_var,
            value="safe-backtest",
        ).place(x=10, y=85)

        # --- Training / model options --------------------------------
        train_frame = ttk.LabelFrame(self, text="Training / Model Options")
        train_frame.place(x=10, y=270, width=395, height=130)

        ttk.Checkbutton(
            train_frame,
            text="Force retrain (ignore existing models)",
            variable=self.force_retrain_var,
        ).place(x=10, y=5)

        ttk.Label(train_frame, text="Models to use:").place(x=10, y=35)
        
        # Row 1 of models
        ttk.Checkbutton(train_frame, text="LSTM", variable=self.models_lstm_var).place(
            x=110, y=33
        )
        ttk.Checkbutton(train_frame, text="GRU", variable=self.models_gru_var).place(
            x=170, y=33
        )
        ttk.Checkbutton(
            train_frame, text="Transformer", variable=self.models_transformer_var
        ).place(x=230, y=33)
        
        # Row 2 of models
        ttk.Checkbutton(train_frame, text="TCN", variable=self.models_tcn_var).place(
            x=110, y=60
        )
        ttk.Checkbutton(train_frame, text="LightGBM", variable=self.models_lgbm_var).place(
            x=170, y=60
        )
        
        # Model info label
        ttk.Label(
            train_frame,
            text="LSTM+Transformer+LightGBM recommended for best results",
            foreground="gray",
            font=("Segoe UI", 7)
        ).place(x=10, y=90)

        # --- Prediction options --------------------------------------
        pred_frame = ttk.LabelFrame(self, text="Prediction Options")
        pred_frame.place(x=415, y=270, width=395, height=130)

        ttk.Checkbutton(
            pred_frame, text="Use Kalman smoothing", variable=self.use_kalman_var
        ).place(x=10, y=5)

        ttk.Label(pred_frame, text="Continuous interval (mins):").place(x=10, y=35)
        ttk.Entry(pred_frame, textvariable=self.interval_var, width=6).place(x=180, y=33)

        ttk.Label(
            pred_frame,
            text=(
                "For continuous mode only. For one-shot prediction,\n"
                "interval is ignored. Kalman smoothing helps reduce\n"
                "prediction noise over time."
            ),
            foreground="gray",
        ).place(x=10, y=60)

        # --- Run / status --------------------------------------------
        action_frame = ttk.Frame(self)
        action_frame.place(x=10, y=405, width=800, height=40)

        ttk.Button(action_frame, text="▶ Run", command=self._on_run_clicked).place(
            x=10, y=5, width=120, height=28
        )
        ttk.Button(action_frame, text="Save MT5 Path", command=self._save_mt5_path).place(
            x=140, y=5, width=120, height=28
        )
        ttk.Button(action_frame, text="Clear Log", command=self._clear_log).place(
            x=270, y=5, width=100, height=28
        )
        ttk.Button(action_frame, text="Exit", command=self.destroy).place(
            x=680, y=5, width=100, height=28
        )

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(action_frame, text="Status:").place(x=400, y=10)
        self.status_label = ttk.Label(action_frame, textvariable=self.status_var, foreground="green")
        self.status_label.place(x=450, y=10)

        # --- Log window ----------------------------------------------
        log_frame = ttk.LabelFrame(self, text=f"Console Output from {SCRIPT_NAME}")
        log_frame.place(x=10, y=450, width=800, height=190)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=9, width=105, state="disabled",
            font=("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _default_script_path(self) -> str:
        """Try to auto-detect the predictor script in the same folder as this GUI."""
        here = os.path.abspath(os.path.dirname(__file__))
        
        # Try multiple possible script names
        candidates = [
            SCRIPT_NAME,
            "unified_predictor_v8_fixed.py",
            "unified_predictor_v8.py",
            "GGTHpredictor2.py",
        ]
        
        for name in candidates:
            candidate = os.path.join(here, name)
            if os.path.isfile(candidate):
                return candidate
        
        # Return default name even if not found
        return os.path.join(here, SCRIPT_NAME)

    def _load_mt5_path(self):
        """Load MT5 path from config.json"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    mt5_path = config.get("mt5_files_path", "")
                    if mt5_path:
                        self.mt5_path_var.set(mt5_path)
        except Exception as e:
            print(f"Warning: Could not load MT5 path from config: {e}")

    def _save_mt5_path(self):
        """Save MT5 path to config.json"""
        mt5_path = self.mt5_path_var.get().strip()

        if not mt5_path:
            messagebox.showwarning("No Path", "Please enter or browse to your MT5 Files directory.")
            return

        if not os.path.exists(mt5_path):
            messagebox.showerror("Invalid Path", f"Directory does not exist:\n{mt5_path}")
            return

        try:
            import json
            config_path = os.path.join(os.path.dirname(__file__), "config.json")

            # Load existing config or create new
            config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)

            # Update MT5 path
            config["mt5_files_path"] = mt5_path
            config["version"] = "2.0"

            # Save
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            messagebox.showinfo("Success", f"MT5 path saved successfully:\n{mt5_path}")
            self._set_status("MT5 path saved")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config:\n{str(e)}")

    def _browse_python(self):
        path = filedialog.askopenfilename(
            title="Select Python executable",
            filetypes=[("Python", "python.exe;pythonw.exe"), ("All files", "*.*")],
        )
        if path:
            self.python_exe_var.set(path)

    def _browse_script(self):
        path = filedialog.askopenfilename(
            title=f"Select {SCRIPT_NAME}",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if path:
            self.script_path_var.set(path)

    def _browse_mt5(self):
        """Browse for MT5 Files directory"""
        path = filedialog.askdirectory(
            title="Select MT5 Files Directory",
            initialdir=os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal")
        )
        if path:
            self.mt5_path_var.set(path)

    def _append_log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        """Clear the log window"""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def _set_status(self, msg: str, color: str = "green"):
        self.status_var.set(msg)
        self.status_label.configure(foreground=color)
        self.update_idletasks()

    # ------------------------------------------------------------------
    # Building the command
    # ------------------------------------------------------------------
    def _build_command(self):
        python_exe = self.python_exe_var.get().strip()
        script_path = self.script_path_var.get().strip()
        symbol = self.symbol_var.get().strip().upper()

        if not python_exe or not os.path.isfile(python_exe):
            raise RuntimeError("Python executable path is invalid.")
        if not script_path or not os.path.isfile(script_path):
            raise RuntimeError(f"Predictor script path is invalid: {script_path}")
        if not symbol:
            raise RuntimeError("Symbol cannot be empty.")

        base_cmd = [python_exe, "-u", script_path]

        # Which real mode?
        action = self.action_var.get()

        # Translate GUI actions into actual CLI modes
        if action in ("train", "train-multitf", "tune"):
            mode = action
        elif action == "predict-mtf-once":
            mode = "predict-multitf"
        elif action == "predict-mtf-cont":
            mode = "predict-multitf"
        elif action == "backtest":
            mode = "backtest"
        elif action == "safe-backtest":
            # This would require adding the safe backtest mode to CLI
            # For now, use regular backtest
            mode = "backtest"
        else:
            raise RuntimeError(f"Unknown action: {action}")

        cmd = base_cmd + [mode, "--symbol", symbol]

        # Models selection (for train / train-multitf / predict modes)
        models = []
        if self.models_lstm_var.get():
            models.append("lstm")
        if self.models_gru_var.get():
            models.append("gru")
        if self.models_transformer_var.get():
            models.append("transformer")
        if self.models_tcn_var.get():
            models.append("tcn")
        if self.models_lgbm_var.get():
            models.append("lgbm")

        # Validate at least one model selected
        if action in ("train", "train-multitf") and not models:
            raise RuntimeError("Please select at least one model type to train.")

        if action in ("train", "train-multitf"):
            if models:
                cmd += ["--models"] + models
            if self.force_retrain_var.get():
                cmd.append("--force")

        if action in ("predict-mtf-once", "predict-mtf-cont"):
            if models:
                cmd += ["--models"] + models
            if not self.use_kalman_var.get():
                cmd.append("--no-kalman")
            if action == "predict-mtf-cont":
                cmd.append("--continuous")
                interval = max(1, int(self.interval_var.get() or 1))
                cmd += ["--interval", str(interval)]

        return cmd

    # ------------------------------------------------------------------
    # Run button
    # ------------------------------------------------------------------
    def _on_run_clicked(self):
        # Verify MT5 path is configured
        mt5_path = self.mt5_path_var.get().strip()
        if not mt5_path:
            messagebox.showwarning(
                "MT5 Path Not Set",
                "Please set the MT5 Files directory path first.\n\n"
                "This is typically located at:\n"
                "C:\\Users\\YourName\\AppData\\Roaming\\MetaQuotes\\Terminal\\HASH\\MQL5\\Files"
            )
            return

        if not os.path.exists(mt5_path):
            messagebox.showerror(
                "Invalid MT5 Path",
                f"The MT5 Files directory does not exist:\n{mt5_path}\n\n"
                "Please update the path."
            )
            return

        # Run in background thread to keep GUI responsive
        thread = threading.Thread(target=self._run_command_thread, daemon=True)
        thread.start()

    def _run_command_thread(self):
        try:
            cmd = self._build_command()
        except Exception as e:
            messagebox.showerror("Configuration error", str(e))
            return

        self._set_status("Running...", "blue")
        self._clear_log()

        self._append_log("=" * 60 + "\n")
        self._append_log(f"GGTH Predictor GUI v{VERSION}\n")
        self._append_log("=" * 60 + "\n\n")
        self._append_log("Executing:\n  " + " ".join(cmd) + "\n\n")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            self._append_log(f"FAILED to start process: {e}\n")
            self._set_status("Failed.", "red")
            return

        # Stream output
        for line in proc.stdout:
            self._append_log(line)

        proc.wait()
        rc = proc.returncode
        if rc == 0:
            self._set_status("Done.", "green")
            self._append_log("\n" + "=" * 60 + "\n")
            self._append_log("✓ Process completed successfully!\n")
        else:
            self._set_status(f"Finished with errors (code {rc}).", "red")
            self._append_log("\n" + "=" * 60 + "\n")
            self._append_log(f"✗ Process exited with code {rc}\n")


if __name__ == "__main__":
    app = GGTHGui()
    app.mainloop()
