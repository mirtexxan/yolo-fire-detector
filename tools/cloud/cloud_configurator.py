"""GUI editor for experiment configurations and runtime overrides."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import (
    DEFAULT_PRESET_RELATIVE,
    GENERATED_DIR_NAME,
    LATEST_CLOUD_CONFIG_RELATIVE,
    LATEST_CLOUD_META_RELATIVE,
    LATEST_LOCAL_CONFIG_RELATIVE,
    LATEST_LOCAL_META_RELATIVE,
    deep_merge,
    list_runtime_configs,
    load_layered_config,
    PRESETS_DIR_NAME,
)

APP_VERSION = "2026.04.02.4"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip("-") or "config"


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def read_optional_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


class ExperimentConfiguratorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Experiment Configurator v{APP_VERSION}")
        self.geometry("980x760")
        self.minsize(900, 680)

        self.available_configs = self._list_source_presets()
        if not self.available_configs:
            raise FileNotFoundError("Nessun preset sorgente trovato in configs/presets/")

        self.runtime_configs = list_runtime_configs(CONFIGS_DIR)
        if not self.runtime_configs:
            raise FileNotFoundError("Nessun runtime override trovato in configs/runtime/")

        self.loaded_config: dict = {}

        self.source_config_var = tk.StringVar(value=self._default_source_config())
        self.runtime_config_var = tk.StringVar(value=self._default_runtime_config())
        self.save_name_var = tk.StringVar(value="custom.yaml")

        self.project_label_var = tk.StringVar()
        self.persistent_root_var = tk.StringVar()
        self.dataset_label_var = tk.StringVar()
        self.num_images_var = tk.StringVar()
        self.image_size_var = tk.StringVar()
        self.negative_ratio_var = tk.StringVar()
        self.train_split_var = tk.StringVar()
        self.dataset_seed_var = tk.StringVar()
        self.force_regenerate_var = tk.BooleanVar(value=False)
        self.training_label_var = tk.StringVar()
        self.model_size_var = tk.StringVar()
        self.weights_var = tk.StringVar()
        self.device_var = tk.StringVar()
        self.require_gpu_var = tk.BooleanVar(value=True)
        self.epochs_var = tk.StringVar()
        self.batch_size_var = tk.StringVar()
        self.training_image_size_var = tk.StringVar()
        self.resume_policy_var = tk.StringVar()

        self._build_ui()
        self.load_selection()

    def _default_source_config(self) -> str:
        for latest_meta_path in [LATEST_LOCAL_META_RELATIVE, LATEST_CLOUD_META_RELATIVE]:
            latest_meta = read_optional_yaml(CONFIGS_DIR / latest_meta_path)
            source_config = str(latest_meta.get("source_config") or "").strip()
            if source_config and source_config in self.available_configs:
                return source_config
        default_preset = DEFAULT_PRESET_RELATIVE.as_posix()
        if default_preset in self.available_configs:
            return default_preset
        return self.available_configs[0]

    def _default_runtime_config(self) -> str:
        for latest_meta_path in [LATEST_LOCAL_META_RELATIVE, LATEST_CLOUD_META_RELATIVE]:
            latest_meta = read_optional_yaml(CONFIGS_DIR / latest_meta_path)
            runtime_config = str(latest_meta.get("runtime_config") or "").strip()
            if runtime_config and runtime_config in self.runtime_configs:
                return runtime_config
        return self.runtime_configs[0]

    def _resolve_latest_targets(self, runtime_name: str) -> tuple[Path, Path, str]:
        lowered = runtime_name.lower()
        if "colab" in lowered or "cloud" in lowered:
            return LATEST_CLOUD_CONFIG_RELATIVE, LATEST_CLOUD_META_RELATIVE, "cloud"
        return LATEST_LOCAL_CONFIG_RELATIVE, LATEST_LOCAL_META_RELATIVE, "local"

    def _list_source_presets(self) -> list[str]:
        preset_root = CONFIGS_DIR / PRESETS_DIR_NAME
        if not preset_root.exists():
            return []
        return sorted(path.relative_to(CONFIGS_DIR).as_posix() for path in preset_root.rglob("*.yaml") if path.is_file())

    def _to_project_relative_path(self, path_value: str) -> str:
        path = Path(path_value).resolve()
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    def _suggest_save_name(self) -> str:
        source_stem = Path(self.source_config_var.get().strip()).stem or "config"
        runtime_stem = Path(self.runtime_config_var.get().strip()).stem or "runtime"
        return f"{slugify(source_stem)}--{slugify(runtime_stem)}.yaml"

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(container)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="Experiment Configurator", font=("Segoe UI", 18, "bold")).pack(side=tk.LEFT)
        ttk.Label(
            header,
            text=f"Version {APP_VERSION}",
            font=("Segoe UI", 10, "bold"),
            foreground="#1f5aa6",
        ).pack(side=tk.RIGHT)

        ttk.Label(
            container,
            text=(
                "Scegli un preset sorgente, applica un runtime override obbligatorio e salva la config finale "
                "in configs/generated/. Gli override avanzati stanno in una tab separata per non allungare la finestra."
            ),
            wraplength=980,
        ).pack(anchor=tk.W, pady=(0, 10))

        notebook = ttk.Notebook(container)
        notebook.pack(fill=tk.BOTH, expand=True)

        base_tab = ttk.Frame(notebook, padding=6)
        advanced_tab = ttk.Frame(notebook, padding=6)
        notebook.add(base_tab, text="Base Config")
        notebook.add(advanced_tab, text="Advanced Overrides")

        source_frame = ttk.LabelFrame(base_tab, text="Preset, runtime e salvataggio", padding=10)
        source_frame.pack(fill=tk.X, pady=(0, 10))
        source_frame.columnconfigure(1, weight=1)
        source_frame.columnconfigure(3, weight=1)

        ttk.Label(source_frame, text="Preset sorgente").grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.source_combo = ttk.Combobox(
            source_frame,
            textvariable=self.source_config_var,
            values=self.available_configs,
            state="readonly",
        )
        self.source_combo.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(source_frame, text="Runtime override").grid(row=0, column=2, sticky=tk.W, padx=(12, 8), pady=4)
        self.runtime_combo = ttk.Combobox(
            source_frame,
            textvariable=self.runtime_config_var,
            values=self.runtime_configs,
            state="readonly",
        )
        self.runtime_combo.grid(row=0, column=3, sticky="ew", pady=4)
        ttk.Button(source_frame, text="Carica selezione", command=self.load_selection).grid(row=0, column=4, padx=(12, 0), pady=4)

        ttk.Label(source_frame, text="Nome file generato").grid(row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        ttk.Entry(source_frame, textvariable=self.save_name_var).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(source_frame, text="Salvato sotto configs/generated/ e aggiornato come latest di ambiente").grid(
            row=1, column=2, columnspan=3, sticky=tk.W, pady=4
        )

        fields_frame = ttk.Frame(base_tab)
        fields_frame.pack(fill=tk.BOTH, expand=True)
        fields_frame.columnconfigure(0, weight=1)
        fields_frame.columnconfigure(1, weight=1)

        self._build_project_dataset_frame(fields_frame)
        self._build_training_frame(fields_frame)
        self._build_overrides_frame(advanced_tab)

        buttons = ttk.Frame(container)
        buttons.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(buttons, text="Ricarica selezione", command=self.load_selection).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Salva config finale + latest ambiente", command=self.save_config).pack(side=tk.RIGHT)

    def _build_project_dataset_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Project + Dataset", padding=10)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 10))
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(10, weight=1)

        rows = [
            ("Project label", self.project_label_var),
            ("Persistent root", self.persistent_root_var),
            ("Dataset label", self.dataset_label_var),
            ("Num images", self.num_images_var),
            ("Image size", self.image_size_var),
            ("Negative ratio", self.negative_ratio_var),
            ("Train split", self.train_split_var),
            ("Dataset seed", self.dataset_seed_var),
        ]
        for index, (label, variable) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=index, column=0, sticky=tk.W, padx=(0, 8), pady=4)
            ttk.Entry(frame, textvariable=variable).grid(row=index, column=1, sticky="ew", pady=4)

        ttk.Checkbutton(frame, text="Force regenerate dataset", variable=self.force_regenerate_var).grid(
            row=len(rows), column=0, columnspan=2, sticky=tk.W, pady=(6, 6)
        )

        ttk.Label(frame, text="Image asset paths (chiave YAML: dataset.fire_image_paths)").grid(
            row=len(rows) + 1, column=0, columnspan=2, sticky=tk.W, pady=(4, 4)
        )
        assets_frame = ttk.Frame(frame)
        assets_frame.grid(row=len(rows) + 2, column=0, columnspan=2, sticky="nsew")
        assets_frame.columnconfigure(0, weight=1)
        assets_frame.rowconfigure(0, weight=1)

        self.fire_image_paths_listbox = tk.Listbox(assets_frame, selectmode=tk.EXTENDED, exportselection=False, height=8)
        self.fire_image_paths_listbox.grid(row=0, column=0, sticky="nsew")

        assets_scrollbar = ttk.Scrollbar(assets_frame, orient=tk.VERTICAL, command=self.fire_image_paths_listbox.yview)
        assets_scrollbar.grid(row=0, column=1, sticky="ns")
        self.fire_image_paths_listbox.configure(yscrollcommand=assets_scrollbar.set)

        asset_buttons = ttk.Frame(assets_frame)
        asset_buttons.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        ttk.Button(asset_buttons, text="Aggiungi file...", command=self.add_fire_image_paths).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(asset_buttons, text="Rimuovi selezionati", command=self.remove_selected_fire_image_paths).pack(fill=tk.X, pady=6)
        ttk.Button(asset_buttons, text="Pulisci lista", command=self.clear_fire_image_paths).pack(fill=tk.X, pady=6)

    def _build_training_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Training", padding=10)
        frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        rows = [
            ("Training label", self.training_label_var),
            ("Model size", self.model_size_var),
            ("Weights", self.weights_var),
            ("Device", self.device_var),
            ("Epochs", self.epochs_var),
            ("Batch size", self.batch_size_var),
            ("Training image size", self.training_image_size_var),
            ("Resume policy", self.resume_policy_var),
        ]
        for index, (label, variable) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=index, column=0, sticky=tk.W, padx=(0, 8), pady=4)
            ttk.Entry(frame, textvariable=variable).grid(row=index, column=1, sticky="ew", pady=4)

        ttk.Checkbutton(frame, text="Require GPU", variable=self.require_gpu_var).grid(
            row=len(rows), column=0, columnspan=2, sticky=tk.W, pady=(6, 6)
        )

    def _build_overrides_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Overrides YAML", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        ttk.Label(frame, text="dataset_settings_overrides").grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        ttk.Label(frame, text="image_transform_overrides").grid(row=0, column=1, sticky=tk.W, pady=(0, 4))
        ttk.Label(frame, text="training_overrides").grid(row=0, column=2, sticky=tk.W, pady=(0, 4))

        self.dataset_overrides_text = ScrolledText(frame, height=14, wrap=tk.NONE)
        self.dataset_overrides_text.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.image_overrides_text = ScrolledText(frame, height=14, wrap=tk.NONE)
        self.image_overrides_text.grid(row=1, column=1, sticky="nsew", padx=6)
        self.training_overrides_text = ScrolledText(frame, height=14, wrap=tk.NONE)
        self.training_overrides_text.grid(row=1, column=2, sticky="nsew", padx=(6, 0))

    def refresh_options(self, *, selected_name: str | None = None) -> None:
        self.available_configs = self._list_source_presets()
        self.runtime_configs = list_runtime_configs(CONFIGS_DIR)
        self.source_combo["values"] = self.available_configs
        self.runtime_combo["values"] = self.runtime_configs
        if selected_name is not None and selected_name in self.available_configs:
            self.source_config_var.set(selected_name)
        if self.runtime_config_var.get().strip() not in self.runtime_configs and self.runtime_configs:
            self.runtime_config_var.set(self.runtime_configs[0])

    def _set_text(self, widget: ScrolledText, content: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert("1.0", content)

    def _set_listbox_items(self, widget: tk.Listbox, items: list[str]) -> None:
        widget.delete(0, tk.END)
        for item in items:
            widget.insert(tk.END, item)

    def _get_listbox_items(self, widget: tk.Listbox) -> list[str]:
        return [str(widget.get(index)).strip() for index in range(widget.size()) if str(widget.get(index)).strip()]

    def add_fire_image_paths(self) -> None:
        initial_dir = PROJECT_ROOT / "base_fire_images"
        file_paths = filedialog.askopenfilenames(
            title="Seleziona image asset paths",
            initialdir=str(initial_dir if initial_dir.exists() else PROJECT_ROOT),
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
        )
        if not file_paths:
            return

        existing_items = self._get_listbox_items(self.fire_image_paths_listbox)
        known = set(existing_items)
        for file_path in file_paths:
            portable = self._to_project_relative_path(file_path)
            if portable not in known:
                self.fire_image_paths_listbox.insert(tk.END, portable)
                known.add(portable)

    def remove_selected_fire_image_paths(self) -> None:
        for index in reversed(self.fire_image_paths_listbox.curselection()):
            self.fire_image_paths_listbox.delete(index)

    def clear_fire_image_paths(self) -> None:
        self.fire_image_paths_listbox.delete(0, tk.END)

    def _get_yaml_dict(self, widget: ScrolledText, label: str) -> dict:
        raw_text = widget.get("1.0", tk.END).strip()
        if not raw_text:
            return {}
        try:
            payload = yaml.safe_load(raw_text) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML non valida in {label}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{label} deve contenere una mappa YAML")
        return payload

    def load_selection(self) -> None:
        source_config_name = self.source_config_var.get().strip()
        runtime_config_name = self.runtime_config_var.get().strip()
        if not source_config_name:
            raise ValueError("Seleziona un preset sorgente")
        if not runtime_config_name:
            raise ValueError("Seleziona un runtime override")

        source_path = CONFIGS_DIR / Path(source_config_name)
        config = deepcopy(load_layered_config(source_path))

        runtime_path = CONFIGS_DIR / Path(runtime_config_name)
        config = deep_merge(config, load_layered_config(runtime_path))

        self.loaded_config = config

        project = config.get("project", {})
        dataset = config.get("dataset", {})
        training = config.get("training", {})

        self.project_label_var.set(str(project.get("label", "")))
        self.persistent_root_var.set(str(project.get("persistent_root", "")))
        self.dataset_label_var.set(str(dataset.get("label", "")))
        self.num_images_var.set(str(dataset.get("num_images", "")))
        self.image_size_var.set(str(dataset.get("image_size", "")))
        self.negative_ratio_var.set(str(dataset.get("negative_ratio", "")))
        self.train_split_var.set(str(dataset.get("train_split", "")))
        self.dataset_seed_var.set(str(dataset.get("seed", "")))
        self.force_regenerate_var.set(bool(dataset.get("force_regenerate", False)))
        self.training_label_var.set(str(training.get("label", "")))
        self.model_size_var.set(str(training.get("model_size", "")))
        self.weights_var.set(str(training.get("weights", "")))
        self.device_var.set(str(training.get("device", "")))
        self.require_gpu_var.set(bool(training.get("require_gpu", True)))
        self.epochs_var.set(str(training.get("epochs", "")))
        self.batch_size_var.set(str(training.get("batch_size", "")))
        self.training_image_size_var.set(str(training.get("image_size", "")))
        self.resume_policy_var.set(str(training.get("resume", "")))

        fire_paths = [str(item) for item in dataset.get("fire_image_paths", [])]
        self._set_listbox_items(self.fire_image_paths_listbox, fire_paths)
        self._set_text(
            self.dataset_overrides_text,
            yaml.safe_dump(config.get("dataset_settings_overrides", {}), sort_keys=False, allow_unicode=False),
        )
        self._set_text(
            self.image_overrides_text,
            yaml.safe_dump(config.get("image_transform_overrides", {}), sort_keys=False, allow_unicode=False),
        )
        self._set_text(
            self.training_overrides_text,
            yaml.safe_dump(config.get("training_overrides", {}), sort_keys=False, allow_unicode=False),
        )

        self.save_name_var.set(self._suggest_save_name())

    def _parse_int(self, value: str, label: str) -> int:
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{label} deve essere un intero") from exc

    def _parse_float(self, value: str, label: str) -> float:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{label} deve essere un numero") from exc

    def build_config(self) -> dict:
        config = deepcopy(self.loaded_config)
        config.setdefault("project", {})
        config.setdefault("dataset", {})
        config.setdefault("training", {})

        runtime_config_name = self.runtime_config_var.get().strip()
        if not runtime_config_name:
            raise ValueError("Serve selezionare un runtime override")

        fire_paths = self._get_listbox_items(self.fire_image_paths_listbox)
        if not fire_paths:
            raise ValueError("Serve almeno un image asset path")

        config["project"]["label"] = self.project_label_var.get().strip()
        config["project"]["persistent_root"] = self.persistent_root_var.get().strip()
        config["dataset"]["label"] = self.dataset_label_var.get().strip()
        config["dataset"]["fire_image_paths"] = fire_paths
        config["dataset"]["num_images"] = self._parse_int(self.num_images_var.get(), "num_images")
        config["dataset"]["image_size"] = self._parse_int(self.image_size_var.get(), "image_size")
        config["dataset"]["negative_ratio"] = self._parse_float(self.negative_ratio_var.get(), "negative_ratio")
        config["dataset"]["train_split"] = self._parse_float(self.train_split_var.get(), "train_split")
        config["dataset"]["seed"] = self._parse_int(self.dataset_seed_var.get(), "dataset_seed")
        config["dataset"]["force_regenerate"] = bool(self.force_regenerate_var.get())

        config["training"]["label"] = self.training_label_var.get().strip()
        config["training"]["model_size"] = self.model_size_var.get().strip()
        weights_value = self.weights_var.get().strip()
        config["training"]["weights"] = weights_value or None
        config["training"]["device"] = self.device_var.get().strip()
        config["training"]["require_gpu"] = bool(self.require_gpu_var.get())
        config["training"]["epochs"] = self._parse_int(self.epochs_var.get(), "epochs")
        config["training"]["batch_size"] = self._parse_int(self.batch_size_var.get(), "batch_size")
        config["training"]["image_size"] = self._parse_int(self.training_image_size_var.get(), "training.image_size")
        config["training"]["resume"] = self.resume_policy_var.get().strip()

        config["dataset_settings_overrides"] = self._get_yaml_dict(self.dataset_overrides_text, "dataset_settings_overrides")
        config["image_transform_overrides"] = self._get_yaml_dict(self.image_overrides_text, "image_transform_overrides")
        config["training_overrides"] = self._get_yaml_dict(self.training_overrides_text, "training_overrides")
        config.pop("extends", None)
        return config

    def save_config(self) -> None:
        try:
            config = self.build_config()
            requested_name = Path(self.save_name_var.get().strip()).name
            if not requested_name:
                requested_name = f"{slugify(config['project']['label'])}.yaml"
            if not requested_name.endswith(".yaml"):
                requested_name += ".yaml"

            target_relative = Path(GENERATED_DIR_NAME) / requested_name
            target_path = CONFIGS_DIR / target_relative
            runtime_name = self.runtime_config_var.get().strip()
            latest_relative, latest_meta_relative, latest_kind = self._resolve_latest_targets(runtime_name)
            latest_path = CONFIGS_DIR / latest_relative
            latest_meta_path = CONFIGS_DIR / latest_meta_relative

            write_yaml(target_path, config)
            write_yaml(latest_path, config)
            write_yaml(
                latest_meta_path,
                {
                    "schema_version": 2,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "source_config": self.source_config_var.get().strip(),
                    "runtime_config": runtime_name,
                    "saved_config": target_relative.as_posix(),
                    "latest_config": latest_relative.as_posix(),
                    "latest_kind": latest_kind,
                    "app_version": APP_VERSION,
                },
            )

            self.refresh_options(selected_name=self.source_config_var.get().strip())

            messagebox.showinfo(
                "Config salvata",
                (
                    f"Config scritta in:\n- {target_path}\n- {latest_path}\n\n"
                    f"Metadata latest {latest_kind}:\n- {latest_meta_path}"
                ),
            )
        except Exception as exc:  # pragma: no cover - GUI feedback path
            messagebox.showerror("Errore salvataggio", str(exc))


def main() -> None:
    app = ExperimentConfiguratorApp()
    app.mainloop()


if __name__ == "__main__":
    main()