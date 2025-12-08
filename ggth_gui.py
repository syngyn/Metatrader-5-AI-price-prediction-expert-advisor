import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext


SCRIPT_NAME = "GGTHpredictor2.py"


class GGTHGui(tk.Tk):
    """
    Simple front-end for GGTHpredictor2.py

    It does NOT re-implement the ML logic. It just calls the existing script with
    the same command-line modes you already have:

      - train
      - train-multitf
      - tune
      - predict
      - predict-multitf
      - backtest
      - backtest-multitf
    """

    def __init__(self):
        super().__init__()
        self.title("GGTH Predictor – Control Panel")
        self.geometry("800x600")
        self.resizable(False, False)

        # State variables
        self.symbol_var = tk.StringVar(value="USDJPY")
        self.action_var = tk.StringVar(value="train-multitf")
        self.force_retrain_var = tk.BooleanVar(value=True)
        self.continuous_var = tk.BooleanVar(value=False)
        self.interval_var = tk.IntVar(value=60)
        self.models_lstm_var = tk.BooleanVar(value=True)
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
        cfg_frame.place(x=10, y=10, width=780, height=120)

        ttk.Label(cfg_frame, text="Python exe:").place(x=10, y=10)
        python_entry = ttk.Entry(cfg_frame, textvariable=self.python_exe_var, width=70)
        python_entry.place(x=90, y=8)
        ttk.Button(cfg_frame, text="Browse...", command=self._browse_python).place(
            x=690, y=6, width=70
        )

        ttk.Label(cfg_frame, text="GGTH script:").place(x=10, y=40)
        script_entry = ttk.Entry(cfg_frame, textvariable=self.script_path_var, width=70)
        script_entry.place(x=90, y=38)
        ttk.Button(cfg_frame, text="Browse...", command=self._browse_script).place(
            x=690, y=36, width=70
        )

        ttk.Label(cfg_frame, text="MT5 Files:").place(x=10, y=70)
        mt5_entry = ttk.Entry(cfg_frame, textvariable=self.mt5_path_var, width=70)
        mt5_entry.place(x=90, y=68)
        ttk.Button(cfg_frame, text="Browse...", command=self._browse_mt5).place(
            x=690, y=66, width=70
        )
        
        ttk.Label(cfg_frame, text="(MT5 Terminal\\...\\MQL5\\Files directory)", 
                  foreground="gray", font=("Segoe UI", 7)).place(x=90, y=90)

        # --- Basic settings -------------------------------------------
        basic_frame = ttk.LabelFrame(self, text="Basic Settings")
        basic_frame.place(x=10, y=135, width=380, height=120)

        ttk.Label(basic_frame, text="Symbol:").place(x=10, y=10)
        ttk.Entry(basic_frame, textvariable=self.symbol_var, width=12).place(x=70, y=8)

        ttk.Label(basic_frame, text="Action:").place(x=10, y=40)
        row_y = 35
        ttk.Radiobutton(
            basic_frame,
            text="Train ALL models (multi-TF)",
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
        mode_frame.place(x=410, y=135, width=380, height=120)

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
            text="Generate backtest predictions (multi-TF)",
            variable=self.action_var,
            value="backtest-multitf",
        ).place(x=10, y=60)

        ttk.Radiobutton(
            mode_frame,
            text="Generate backtest predictions (single config)",
            variable=self.action_var,
            value="backtest",
        ).place(x=10, y=85)

        # --- Training / model options --------------------------------
        train_frame = ttk.LabelFrame(self, text="Training / Model Options")
        train_frame.place(x=10, y=260, width=380, height=110)

        ttk.Checkbutton(
            train_frame,
            text="Force retrain (ignore existing models)",
            variable=self.force_retrain_var,
        ).place(x=10, y=5)

        ttk.Label(train_frame, text="Models to use:").place(x=10, y=35)
        ttk.Checkbutton(train_frame, text="LSTM", variable=self.models_lstm_var).place(
            x=120, y=33
        )
        ttk.Checkbutton(
            train_frame, text="Transformer", variable=self.models_transformer_var
        ).place(x=180, y=33)
        ttk.Checkbutton(train_frame, text="TCN", variable=self.models_tcn_var).place(
            x=120, y=60
        )
        ttk.Checkbutton(train_frame, text="LightGBM", variable=self.models_lgbm_var).place(
            x=180, y=60
        )

        # --- Prediction options --------------------------------------
        pred_frame = ttk.LabelFrame(self, text="Prediction Options")
        pred_frame.place(x=410, y=260, width=380, height=110)

        ttk.Checkbutton(
            pred_frame, text="Use Kalman smoothing", variable=self.use_kalman_var
        ).place(x=10, y=5)

        ttk.Label(pred_frame, text="Continuous interval (mins):").place(x=10, y=35)
        ttk.Entry(pred_frame, textvariable=self.interval_var, width=6).place(x=180, y=33)

        ttk.Label(
            pred_frame,
            text=(
                "For continuous mode only. For one-shot prediction,\n"
                "interval is ignored."
            ),
            foreground="gray",
        ).place(x=10, y=60)

        # --- Run / status --------------------------------------------
        action_frame = ttk.Frame(self)
        action_frame.place(x=10, y=375, width=780, height=40)

        ttk.Button(action_frame, text="Run", command=self._on_run_clicked).place(
            x=10, y=5, width=120, height=28
        )
        ttk.Button(action_frame, text="Save MT5 Path", command=self._save_mt5_path).place(
            x=140, y=5, width=120, height=28
        )
        ttk.Button(action_frame, text="Exit", command=self.destroy).place(
            x=650, y=5, width=120, height=28
        )

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(action_frame, text="Status:").place(x=280, y=10)
        ttk.Label(action_frame, textvariable=self.status_var, foreground="green").place(
            x=330, y=10
        )

        # --- Log window ----------------------------------------------
        log_frame = ttk.LabelFrame(self, text="Console Output from GGTHpredictor2.py")
        log_frame.place(x=10, y=415, width=780, height=175)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=8, width=100, state="disabled"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _default_script_path(self) -> str:
        """Try to auto-detect GGTHpredictor2.py in the same folder as this GUI."""
        here = os.path.abspath(os.path.dirname(__file__))
        candidate = os.path.join(here, SCRIPT_NAME)
        if os.path.isfile(candidate):
            return candidate
        return candidate  # still a reasonable default

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
            config["version"] = "1.2"
            
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
            title="Select GGTHpredictor2.py",
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

    def _set_status(self, msg: str):
        self.status_var.set(msg)
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
            raise RuntimeError("GGTHpredictor2.py path is invalid.")
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
        elif action == "backtest-multitf":
            # Script uses 'backtest' for historical prediction generation
            mode = "backtest"
        elif action == "backtest":
            mode = "backtest"
        else:
            raise RuntimeError(f"Unknown action: {action}")

        cmd = base_cmd + [mode, "--symbol", symbol]

        # Models selection (for train / train-multitf / predict modes)
        models = []
        if self.models_lstm_var.get():
            models.append("lstm")
        if self.models_transformer_var.get():
            models.append("transformer")
        if self.models_tcn_var.get():
            models.append("tcn")
        if self.models_lgbm_var.get():
            models.append("lgbm")

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
            # else: one-shot prediction, no extra flags

        # Backtest doesn't take extra args in v7.15, so nothing more to add.

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

        self._set_status("Running...")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

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
            self._set_status("Failed.")
            return

        # Stream output
        for line in proc.stdout:
            self._append_log(line)

        proc.wait()
        rc = proc.returncode
        if rc == 0:
            self._set_status("Done.")
        else:
            self._set_status(f"Finished with errors (code {rc}).")

        self._append_log(f"\nProcess exited with code {rc}\n")


if __name__ == "__main__":
    app = GGTHGui()
    app.mainloop()
