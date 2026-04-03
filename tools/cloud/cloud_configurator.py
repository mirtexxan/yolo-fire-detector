"""GUI editor for experiment configurations and runtime overrides."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_utils import (
    GENERATED_DIR_NAME,
    LATEST_CLOUD_CONFIG_RELATIVE,
    LATEST_CLOUD_META_RELATIVE,
    LATEST_LOCAL_CONFIG_RELATIVE,
    LATEST_LOCAL_META_RELATIVE,
    load_layered_config,
    PRESETS_DIR_NAME,
)
from settings import DatasetGenerationSettings, ImageTransformSettings, TrainingSettings

APP_VERSION = "2026.04.03.14"
CONFIGS_DIR = PROJECT_ROOT / "configs"

LOCAL_PERSISTENT_ROOT_PREFIX = "artifacts/"
CLOUD_PERSISTENT_ROOT_PREFIX = "/content/drive/MyDrive/"
DATASET_PRESETS_DIR = CONFIGS_DIR / PRESETS_DIR_NAME / "dataset"
TRAINING_PRESETS_DIR = CONFIGS_DIR / PRESETS_DIR_NAME / "training"


FIELD_HELP_TEXTS: dict[str, str] = {
    "Config completa": "Carica una configurazione salvata in precedenza per riprendere da li'.",
    "Preset dataset": "Profilo di generazione preconfigurato. Cambia quante immagini generare, la scala del fuoco, le augmentations.",
    "Preset training": "Profilo di training preconfigurato. Cambia modello, epoche e iperparametri.",
    "Ambiente target": "'local' = salva in artifacts/local (PC). 'cloud' = salva in artifacts/cloud (la persistent_root viene sovrascritta dal notebook Colab).",
    "Nome file generato": "Nome del file scritto in configs/generated/. Sceglilo tu.",
    "Project label": "Identificativo esperimento auto-generato da label dataset + label training. Compare nei nomi run/export.",
    "Dataset label": "Nome logico del dataset, compare nella cartella dataset generata.",
    "Num images": "Numero totale di immagini sintetiche da generare.",
    "Image size (dataset)": "Dimensione in pixel delle immagini generate.",
    "Negative ratio": "Quota immagini senza fuoco: 0.35 = 35% senza fuoco.",
    "Train split": "Percentuale training: 0.8 = 80% train, 20% validation.",
    "Dataset seed": "Seed per riproducibilita'. Stesso seed = stesso dataset.",
    "Training label": "Etichetta del training, compare nel run e nell'export.",
    "Model size": "Taglia YOLO descrittiva: Nano, Small, Medium, Large, XLarge.",
    "Device": "Dove eseguire: gpu = GPU, cpu = CPU.",
    "Epochs": "Numero di epoche di addestramento.",
    "Batch size": "Immagini per step. Riduci se esaurisci la VRAM.",
    "Image size (training)": "Risoluzione usata da YOLO durante il training.",
    "Resume policy": "auto = riprende da checkpoint se esiste, never = ricomincia sempre.",
    "Force regenerate dataset": "Se attivo, rigenera il dataset anche se una versione compatibile esiste gia'.",
    "Require GPU": "Se attivo, blocca il training se CUDA non e' disponibile (evita training CPU lento involontario).",
}

MODEL_SIZE_DISPLAY_TO_CODE: dict[str, str] = {
    "Nano (veloce)": "n",
    "Small (bilanciato)": "s",
    "Medium (piu accurato)": "m",
    "Large (alto costo)": "l",
    "XLarge (massima capacita)": "x",
}
MODEL_SIZE_CODE_TO_DISPLAY: dict[str, str] = {code: label for label, code in MODEL_SIZE_DISPLAY_TO_CODE.items()}
DEVICE_DISPLAY_TO_CONFIG: dict[str, str] = {
    "gpu": "0",
    "cpu": "cpu",
}
DEVICE_CONFIG_TO_DISPLAY: dict[str, str] = {
    "0": "gpu",
    "cpu": "cpu",
}

ADVANCED_FIELD_HELP_TEXTS: dict[str, str] = {
    "fire_scale_min": "Dimensione minima del fuoco nell'immagine. Valori bassi = fuochi piu' piccoli.",
    "fire_scale_max": "Dimensione massima del fuoco. Valori alti = fuochi grandi e vicini.",
    "rotation_deg_min": "Rotazione minima applicata al fuoco (gradi).",
    "rotation_deg_max": "Rotazione massima applicata al fuoco (gradi).",
    "perspective_shift": "Intensita' distorsione prospettica. Valori alti = piu' varieta', piu' difficolta'.",
    "enable_color_shift": "Attiva variazioni colore del fuoco (simula condizioni diverse).",
    "color_shift_prob": "Probabilita' di applicare variazione colore (0.0-1.0).",
    "motion_blur_prob": "Probabilita' di blur da movimento (0.0-1.0).",
    "gaussian_blur_prob": "Probabilita' di blur gaussiano (0.0-1.0).",
    "noise_prob": "Probabilita' di aggiungere rumore visivo (0.0-1.0).",
    "shadow_prob": "Probabilita' di ombre sintetiche vicino al fuoco (0.0-1.0).",
    "occlusion_prob": "Probabilita' di occlusioni parziali del fuoco (0.0-1.0).",
    "augment_negative_backgrounds": "Augmentazione sulle immagini negative, riduce falsi positivi.",
    "patience": "Epoche senza miglioramento prima dello stop anticipato.",
    "learning_rate_init": "Learning rate iniziale.",
    "learning_rate_final": "Learning rate finale.",
    "momentum": "Momentum dell'ottimizzatore.",
    "weight_decay": "Regolarizzazione L2 (riduce overfitting).",
    "rotation_degrees": "Rotazione per augmentazioni YOLO in training.",
    "translate": "Traslazione massima in training YOLO.",
    "scale": "Scala massima in training YOLO.",
    "flip_vertical": "Probabilita' flip verticale.",
    "flip_horizontal": "Probabilita' flip orizzontale.",
    "mosaic": "Intensita' mosaic augmentation YOLO.",
    "mixed_precision": "Precisione mista (piu' veloce su GPU compatibili).",
}


class HoverToolTip:
    """Minimal tooltip shown on widget hover to keep the UI clean."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text.strip()
        self.tip_window: tk.Toplevel | None = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)

    def _show(self, _event: tk.Event) -> None:
        if not self.text or self.tip_window is not None:
            return

        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8

        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tip_window,
            text=self.text,
            justify=tk.LEFT,
            relief=tk.SOLID,
            borderwidth=1,
            background="#fffde7",
            foreground="#111111",
            padx=8,
            pady=6,
            wraplength=420,
        )
        label.pack()

    def _hide(self, _event: tk.Event) -> None:
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


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
        self._fit_window_to_screen()

        self.available_dataset_presets = self._list_dataset_presets()
        self.available_training_presets = self._list_training_presets()
        self.available_complete_configs = self._list_complete_configs()
        self.loaded_config: dict = self._new_base_config()

        self.complete_config_var = tk.StringVar(value="")
        self.dataset_preset_var = tk.StringVar(value="")
        self.training_preset_var = tk.StringVar(value="")
        self.target_env_var = tk.StringVar(value="local")
        self.save_name_var = tk.StringVar(value="config.yaml")

        self.project_label_preview_var = tk.StringVar(value="fire-project")
        self.dataset_label_var = tk.StringVar()
        self.num_images_var = tk.StringVar()
        self.image_size_var = tk.StringVar()
        self.negative_ratio_var = tk.StringVar()
        self.train_split_var = tk.StringVar()
        self.dataset_seed_var = tk.StringVar()
        self.force_regenerate_var = tk.BooleanVar(value=False)
        self.training_label_var = tk.StringVar()
        self.model_size_var = tk.StringVar()
        self.device_var = tk.StringVar()
        self.require_gpu_var = tk.BooleanVar(value=True)
        self.epochs_var = tk.StringVar()
        self.batch_size_var = tk.StringVar()
        self.training_image_size_var = tk.StringVar()
        self.resume_policy_var = tk.StringVar()
        self.base_readiness_var = tk.StringVar(value="Base readiness: pending")
        self._syncing_image_size = False
        self._auto_save_name_enabled = True
        self._updating_save_name_programmatically = False
        self._last_auto_save_name = "config.yaml"

        self.override_specs = self._collect_override_specs()
        self.guided_override_enabled_vars: dict[str, dict[str, tk.BooleanVar]] = {
            section: {} for section in self.override_specs
        }
        self.guided_override_value_vars: dict[str, dict[str, tk.StringVar]] = {
            section: {} for section in self.override_specs
        }
        self.guided_original_value_vars: dict[str, dict[str, tk.StringVar]] = {
            section: {} for section in self.override_specs
        }

        self._build_ui()
        for var in [
            self.dataset_label_var,
            self.num_images_var,
            self.image_size_var,
            self.negative_ratio_var,
            self.train_split_var,
            self.dataset_seed_var,
            self.training_label_var,
            self.model_size_var,
            self.device_var,
            self.epochs_var,
            self.batch_size_var,
            self.training_image_size_var,
            self.resume_policy_var,
        ]:
            var.trace_add("write", self._on_base_field_change)
        self.dataset_label_var.trace_add("write", self._update_project_label_preview)
        self.training_label_var.trace_add("write", self._update_project_label_preview)
        self.image_size_var.trace_add("write", self._sync_dataset_to_training_image_size)
        self.training_image_size_var.trace_add("write", self._sync_training_to_dataset_image_size)
        self.save_name_var.trace_add("write", self._on_save_name_change)
        self._apply_config_to_form(self.loaded_config)

    def _list_dataset_presets(self) -> list[str]:
        if not DATASET_PRESETS_DIR.exists():
            return []
        return sorted(p.stem for p in DATASET_PRESETS_DIR.glob("*.yaml") if p.is_file())

    def _list_training_presets(self) -> list[str]:
        if not TRAINING_PRESETS_DIR.exists():
            return []
        return sorted(p.stem for p in TRAINING_PRESETS_DIR.glob("*.yaml") if p.is_file())

    def _list_complete_configs(self) -> list[str]:
        root = CONFIGS_DIR / GENERATED_DIR_NAME
        if not root.exists():
            return []
        return sorted(p.relative_to(CONFIGS_DIR).as_posix() for p in root.rglob("*.yaml") if p.is_file() and not p.name.endswith(".meta.yaml"))

    def _new_base_config(self) -> dict:
        fire_paths: list[str] = []
        raw_paths = getattr(DatasetGenerationSettings, "FIRE_IMAGE_PATHS", [])
        if isinstance(raw_paths, list):
            fire_paths = [str(item).replace("\\", "/") for item in raw_paths]

        return {
            "project": {
                "label": "fire-project",
                "persistent_root": "artifacts/local",
            },
            "dataset": {
                "label": "dataset-name",
                "fire_image_paths": [],
                "num_images": None,
                "image_size": None,
                "negative_ratio": None,
                "train_split": None,
                "seed": None,
                "force_regenerate": False,
            },
            "training": {
                "label": "training-name",
                "model_size": "",
                "device": "",
                "require_gpu": True,
                "epochs": None,
                "batch_size": 16,
                "image_size": None,
                "resume": "",
            },
            "dataset_settings_overrides": {},
            "image_transform_overrides": {},
            "training_overrides": {},
        }

    def _to_project_relative_path(self, path_value: str) -> str:
        path = Path(path_value).resolve()
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    def _compute_persistent_root(self) -> str:
        env = self.target_env_var.get().strip().lower()
        if env == "cloud":
            return f"{CLOUD_PERSISTENT_ROOT_PREFIX}yolo-fire-detector"
        return "artifacts/local"

    def _on_base_field_change(self, *_args: object) -> None:
        self._update_base_checklist()

    def _update_project_label_preview(self, *_args: object) -> None:
        dataset_label = self.dataset_label_var.get().strip()
        training_label = self.training_label_var.get().strip()
        if dataset_label and training_label:
            self.project_label_preview_var.set(slugify(f"{dataset_label}-{training_label}"))
        elif dataset_label:
            self.project_label_preview_var.set(slugify(dataset_label))
        elif training_label:
            self.project_label_preview_var.set(slugify(training_label))
        else:
            self.project_label_preview_var.set("fire-project")
        self._update_auto_save_name_from_experiment()

    def _sync_dataset_to_training_image_size(self, *_args: object) -> None:
        if self._syncing_image_size:
            return
        self._syncing_image_size = True
        self.training_image_size_var.set(self.image_size_var.get())
        self._syncing_image_size = False

    def _sync_training_to_dataset_image_size(self, *_args: object) -> None:
        if self._syncing_image_size:
            return
        self._syncing_image_size = True
        self.image_size_var.set(self.training_image_size_var.get())
        self._syncing_image_size = False

    def _to_model_size_code(self, display_value: str) -> str:
        text = display_value.strip()
        if text in MODEL_SIZE_DISPLAY_TO_CODE:
            return MODEL_SIZE_DISPLAY_TO_CODE[text]
        return text

    def _to_model_size_display(self, code_value: str) -> str:
        text = code_value.strip()
        if text in MODEL_SIZE_DISPLAY_TO_CODE:
            return text
        code = text.lower()
        return MODEL_SIZE_CODE_TO_DISPLAY.get(code, "")

    def _to_device_config(self, display_value: str) -> str:
        text = display_value.strip().lower()
        return DEVICE_DISPLAY_TO_CONFIG.get(text, text)

    def _to_device_display(self, config_value: str) -> str:
        text = config_value.strip().lower()
        return DEVICE_CONFIG_TO_DISPLAY.get(text, text)

    def _on_save_name_change(self, *_args: object) -> None:
        if self._updating_save_name_programmatically:
            return
        if self.save_name_var.get().strip() != self._last_auto_save_name:
            self._auto_save_name_enabled = False

    def _update_auto_save_name_from_experiment(self) -> None:
        if not self._auto_save_name_enabled:
            return
        auto_name = f"{self.project_label_preview_var.get().strip() or 'config'}.yaml"
        self._last_auto_save_name = auto_name
        self._updating_save_name_programmatically = True
        self.save_name_var.set(auto_name)
        self._updating_save_name_programmatically = False

    def _validate_numeric_input(self, text: str, allow_decimal: bool) -> bool:
        if text == "":
            return True
        if allow_decimal:
            return bool(re.fullmatch(r"[0-9]*(\.[0-9]*)?", text))
        return bool(re.fullmatch(r"[0-9]+", text))

    def _bind_numeric_validation(self, entry: ttk.Entry, *, allow_decimal: bool) -> None:
        vcmd = (self.register(lambda text, dec=allow_decimal: self._validate_numeric_input(text, dec)), "%P")
        entry.configure(validate="key", validatecommand=vcmd)

    def _is_positive_int(self, value: str) -> bool:
        try:
            return int(value.strip()) > 0
        except Exception:
            return False

    def _is_ratio(self, value: str) -> bool:
        try:
            parsed = float(value.strip())
            return 0.0 <= parsed <= 1.0
        except Exception:
            return False

    def _update_base_checklist(self) -> None:
        images_ok = len(self._get_listbox_items(self.fire_image_paths_listbox)) > 0 if hasattr(self, "fire_image_paths_listbox") else False
        dataset_ok = (
            bool(self.dataset_label_var.get().strip())
            and self._is_positive_int(self.num_images_var.get())
            and self._is_positive_int(self.image_size_var.get())
            and self._is_ratio(self.negative_ratio_var.get())
            and self._is_ratio(self.train_split_var.get())
            and self._is_positive_int(self.dataset_seed_var.get())
        )
        training_ok = (
            bool(self.training_label_var.get().strip())
            and self._to_model_size_code(self.model_size_var.get()) in {"n", "s", "m", "l", "x"}
            and self.device_var.get().strip().lower() in {"gpu", "cpu"}
            and self._is_positive_int(self.epochs_var.get())
            and self._is_positive_int(self.training_image_size_var.get())
            and self.resume_policy_var.get().strip() in {"auto", "never"}
        )

        text = (
            f"Checklist base: immagini=[{'OK' if images_ok else 'MISSING'}]  "
            f"dataset=[{'OK' if dataset_ok else 'MISSING'}]  "
            f"training=[{'OK' if training_ok else 'MISSING'}]"
        )
        self.base_readiness_var.set(text)
        if hasattr(self, "base_readiness_label"):
            self.base_readiness_label.configure(fg="#1f7a1f" if (images_ok and dataset_ok and training_ok) else "#a15c00")
        if hasattr(self, "save_button_top"):
            self.save_button_top.configure(state=(tk.NORMAL if (images_ok and dataset_ok and training_ok) else tk.DISABLED))

    def _resolve_latest_targets(self, target_env: str) -> tuple[Path, Path, str]:
        if target_env.strip().lower() == "cloud":
            return LATEST_CLOUD_CONFIG_RELATIVE, LATEST_CLOUD_META_RELATIVE, "cloud"
        return LATEST_LOCAL_CONFIG_RELATIVE, LATEST_LOCAL_META_RELATIVE, "local"

    # ------------------------------------------------------------------ presets

    def _load_preset_yaml(self, path: Path) -> dict:
        if not path.exists():
            raise ValueError(f"Preset non trovato: {path.name}")
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def apply_dataset_preset(self) -> None:
        name = self.dataset_preset_var.get().strip()
        if not name:
            return
        try:
            data = self._load_preset_yaml(DATASET_PRESETS_DIR / f"{name}.yaml")
            ds = data.get("dataset", {})
            for key, var in [
                ("num_images", self.num_images_var),
                ("image_size", self.image_size_var),
                ("negative_ratio", self.negative_ratio_var),
                ("train_split", self.train_split_var),
                ("seed", self.dataset_seed_var),
            ]:
                if key in ds:
                    var.set(str(ds[key]))
            if "force_regenerate" in ds:
                self.force_regenerate_var.set(bool(ds["force_regenerate"]))
            for section in ("dataset_settings_overrides", "image_transform_overrides"):
                section_data = data.get(section, {})
                if isinstance(section_data, dict):
                    for field_name, orig_var in self.guided_original_value_vars[section].items():
                        orig_var.set(self._value_to_text(section_data[field_name]) if field_name in section_data else "(non nel preset)")
        except Exception as exc:
            messagebox.showerror("Errore preset dataset", str(exc))

    def apply_training_preset(self) -> None:
        name = self.training_preset_var.get().strip()
        if not name:
            return
        try:
            data = self._load_preset_yaml(TRAINING_PRESETS_DIR / f"{name}.yaml")
            tr = data.get("training", {})
            for key, var in [
                ("model_size", self.model_size_var),
                ("epochs", self.epochs_var),
                ("batch_size", self.batch_size_var),
                ("image_size", self.training_image_size_var),
                ("resume", self.resume_policy_var),
            ]:
                if key in tr:
                    if key == "model_size":
                        var.set(self._to_model_size_display(str(tr[key])))
                    else:
                        var.set(str(tr[key]))
            section_data = data.get("training_overrides", {})
            if isinstance(section_data, dict):
                for field_name, orig_var in self.guided_original_value_vars["training_overrides"].items():
                    orig_var.set(self._value_to_text(section_data[field_name]) if field_name in section_data else "(non nel preset)")
        except Exception as exc:
            messagebox.showerror("Errore preset training", str(exc))

    # --------------------------------------------------------- load / refresh

    def load_complete_config(self) -> None:
        config_name = self.complete_config_var.get().strip()
        if not config_name:
            messagebox.showwarning("Config completa", "Seleziona prima una config dall'elenco.")
            return
        try:
            config_path = CONFIGS_DIR / Path(config_name)
            if not config_path.exists():
                raise ValueError(f"Config non trovata: {config_name}")
            config = deepcopy(load_layered_config(config_path))
            self.loaded_config = config
            self._apply_config_to_form(config)
        except Exception as exc:
            messagebox.showerror("Errore caricamento", str(exc))

    def load_blank(self) -> None:
        self.complete_config_var.set("")
        self.dataset_preset_var.set("")
        self.training_preset_var.set("")
        self.loaded_config = self._new_base_config()
        self._apply_config_to_form(self.loaded_config)

    def refresh_options(self) -> None:
        self.available_complete_configs = self._list_complete_configs()
        self.complete_config_combo["values"] = self.available_complete_configs

    # --------------------------------------------------------------- build UI

    def _fit_window_to_screen(self) -> None:
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        target_w = min(900, max(1040, screen_w - 80))
        target_h = min(900, max(760, screen_h - 80))
        self.geometry(f"{target_w}x{target_h}")
        self.minsize(980, 720)

    def _add_help_tooltip(self, widget: tk.Widget, text: str | None) -> None:
        if text and text.strip():
            HoverToolTip(widget, text)

    def _label_help(self, label_text: str) -> str:
        return FIELD_HELP_TEXTS.get(label_text, f"Campo '{label_text}'.")

    def _advanced_field_help(self, field_name: str, default_value: object) -> str:
        base_text = ADVANCED_FIELD_HELP_TEXTS.get(
            field_name,
            "Opzione avanzata. Attivala solo se vuoi cambiare il comportamento rispetto ai valori standard.",
        )
        return f"{base_text}\nValore standard: {self._value_to_text(default_value)}"

    def _build_ui(self) -> None:
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Blue.TNotebook.Tab", padding=[12, 7, 12, 7], font=("Segoe UI", 10, "bold"))
        style.map(
            "Blue.TNotebook.Tab",
            padding=[("selected", [12, 7, 12, 7]), ("!selected", [12, 7, 12, 7])],
            background=[("selected", "#1f6feb"), ("active", "#dbeafe")],
            foreground=[("selected", "#ffffff"), ("!selected", "#0b3a75")],
        )

        # --- fixed top header ---
        top_frame = ttk.Frame(container, padding=(10, 6, 10, 3))
        top_frame.pack(fill=tk.X)

        header = ttk.Frame(top_frame)
        header.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(header, text="Experiment Configurator", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=(0, 8))

        save_name_row = ttk.Frame(header)
        save_name_row.pack(side=tk.RIGHT)
        env_switch = ttk.Frame(save_name_row)
        env_switch.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(env_switch, text="Ambiente:").pack(side=tk.LEFT)
        ttk.Radiobutton(env_switch, text="Local", variable=self.target_env_var, value="local").pack(side=tk.LEFT, padx=(4, 2))
        ttk.Radiobutton(env_switch, text="Cloud", variable=self.target_env_var, value="cloud").pack(side=tk.LEFT, padx=(2, 0))
        ttk.Label(save_name_row, text="File config:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(save_name_row, textvariable=self.save_name_var, width=28).pack(side=tk.LEFT, padx=(0, 6))

        self.save_button_top = tk.Button(
            save_name_row,
            text="Salva config finale",
            command=self.save_config,
            bg="#1f6feb", fg="#ffffff",
            activebackground="#1959bb", activeforeground="#ffffff",
            relief=tk.FLAT, padx=10, pady=4,
            font=("Segoe UI", 9, "bold"),
        )
        self.save_button_top.pack(side=tk.LEFT)
        self._add_help_tooltip(self.save_button_top, "Salva la configurazione in configs/generated/ e aggiorna il latest dell'ambiente selezionato.")

        # --- body without global page scrolling ---
        self.main_content = ttk.Frame(container, padding=8)
        self.main_content.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(self.main_content, style="Blue.TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_generale = ttk.Frame(notebook, padding=8)
        tab_presets = ttk.Frame(notebook, padding=8)
        tab_advanced = ttk.Frame(notebook, padding=8)
        notebook.add(tab_generale, text="Generale")
        notebook.add(tab_presets, text="Presets")
        notebook.add(tab_advanced, text="Avanzate")

        self._build_generale_tab(tab_generale)
        self._build_presets_tab(tab_presets)
        self._build_advanced_tab(tab_advanced)

    # ------------------------------------------------------ tab Generale

    def _build_generale_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        # Esperimento auto-generato
        proj_frame = ttk.LabelFrame(parent, text="Identificativo esperimento (auto)", padding=10)
        proj_frame.pack(fill=tk.X, pady=(0, 10))
        proj_frame.columnconfigure(1, weight=1)
        proj_label = ttk.Label(proj_frame, text="ID run/export")
        proj_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self._add_help_tooltip(proj_label, self._label_help("Project label"))
        ttk.Entry(proj_frame, textvariable=self.project_label_preview_var, state="readonly").grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(
            proj_frame,
            text="Generato automaticamente da Dataset label + Training label.",
            foreground="#555555",
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W)

        # Base obbligatoria: dataset + training
        base_frame = ttk.LabelFrame(parent, text="Scelte base obbligatorie", padding=10)
        base_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            base_frame,
            text="Compila qui i parametri minimi prima di salvare: immagini di base, opzioni dataset e opzioni training.",
            foreground="#555555",
            wraplength=900,
        ).pack(anchor=tk.W, pady=(0, 8))
        self.base_readiness_label = tk.Label(base_frame, textvariable=self.base_readiness_var, anchor="w", fg="#a15c00")
        self.base_readiness_label.pack(fill=tk.X, pady=(0, 8))

        img_frame = ttk.LabelFrame(base_frame, text="Immagini di fuoco di base  ★ obbligatorie", padding=10)
        img_frame.pack(fill=tk.X, pady=(0, 10))
        img_frame.columnconfigure(0, weight=1)
        img_frame.rowconfigure(0, weight=1)
        listbox_row = ttk.Frame(img_frame)
        listbox_row.pack(fill=tk.BOTH, expand=True)
        listbox_row.columnconfigure(0, weight=1)
        self.fire_image_paths_listbox = tk.Listbox(listbox_row, selectmode=tk.EXTENDED, exportselection=False, height=6)
        self.fire_image_paths_listbox.grid(row=0, column=0, sticky="nsew")
        assets_scrollbar = ttk.Scrollbar(listbox_row, orient=tk.VERTICAL, command=self.fire_image_paths_listbox.yview)
        assets_scrollbar.grid(row=0, column=1, sticky="ns")
        self.fire_image_paths_listbox.configure(yscrollcommand=assets_scrollbar.set)
        asset_buttons = ttk.Frame(listbox_row)
        asset_buttons.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        ttk.Button(asset_buttons, text="Aggiungi file...", command=self.add_fire_image_paths).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(asset_buttons, text="Rimuovi selezionati", command=self.remove_selected_fire_image_paths).pack(fill=tk.X, pady=4)
        ttk.Button(asset_buttons, text="Pulisci lista", command=self.clear_fire_image_paths).pack(fill=tk.X, pady=4)
        ttk.Label(
            img_frame,
            text="I file PNG/JPG con sfondo trasparente funzionano meglio. Usa base_fire_images/ come punto di partenza.",
            foreground="#555555",
            wraplength=900,
        ).pack(anchor=tk.W, pady=(6, 0))

        ds_frame = ttk.LabelFrame(base_frame, text="Opzioni base dataset", padding=10)
        ds_frame.pack(fill=tk.X, pady=(0, 10))
        ds_frame.columnconfigure(1, weight=1)
        ds_frame.columnconfigure(3, weight=1)
        dataset_fields = [
            ("Dataset label", self.dataset_label_var, "Dataset label"),
            ("Num images", self.num_images_var, "Num images"),
            ("Image size (dataset)", self.image_size_var, "Image size (dataset)"),
            ("Negative ratio", self.negative_ratio_var, "Negative ratio"),
            ("Train split", self.train_split_var, "Train split"),
            ("Dataset seed", self.dataset_seed_var, "Dataset seed"),
        ]
        for idx, (txt, var, help_key) in enumerate(dataset_fields):
            row, col = divmod(idx, 2)
            col_label = col * 2
            lbl = ttk.Label(ds_frame, text=txt)
            lbl.grid(row=row, column=col_label, sticky=tk.W, padx=(0 if col_label == 0 else 10, 8), pady=4)
            self._add_help_tooltip(lbl, self._label_help(help_key))
            entry = ttk.Entry(ds_frame, textvariable=var)
            entry.grid(row=row, column=col_label + 1, sticky="ew", pady=4)
            if txt in {"Num images", "Image size (dataset)", "Dataset seed"}:
                self._bind_numeric_validation(entry, allow_decimal=False)
            elif txt in {"Negative ratio", "Train split"}:
                self._bind_numeric_validation(entry, allow_decimal=True)
            if txt == "Dataset label":
                entry.insert(0, "dataset-name")
        force_cb = ttk.Checkbutton(ds_frame, text="Force regenerate dataset", variable=self.force_regenerate_var)
        force_cb.grid(row=(len(dataset_fields) + 1) // 2, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))
        self._add_help_tooltip(force_cb, self._label_help("Force regenerate dataset"))

        tr_frame = ttk.LabelFrame(base_frame, text="Opzioni base training", padding=10)
        tr_frame.pack(fill=tk.X, pady=(0, 10))
        tr_frame.columnconfigure(1, weight=1)
        tr_frame.columnconfigure(3, weight=1)
        training_fields = [
            ("Training label", self.training_label_var, "Training label"),
            ("Model size", self.model_size_var, "Model size"),
            ("Device", self.device_var, "Device"),
            ("Epochs", self.epochs_var, "Epochs"),
            ("Image size (training)", self.training_image_size_var, "Image size (training)"),
            ("Resume policy", self.resume_policy_var, "Resume policy"),
        ]
        for idx, (txt, var, help_key) in enumerate(training_fields):
            row, col = divmod(idx, 2)
            col_label = col * 2
            lbl = ttk.Label(tr_frame, text=txt)
            lbl.grid(row=row, column=col_label, sticky=tk.W, padx=(0 if col_label == 0 else 10, 8), pady=4)
            self._add_help_tooltip(lbl, self._label_help(help_key))
            if txt == "Model size":
                ttk.Combobox(
                    tr_frame,
                    textvariable=var,
                    values=list(MODEL_SIZE_DISPLAY_TO_CODE.keys()),
                    state="readonly",
                ).grid(row=row, column=col_label + 1, sticky="ew", pady=4)
            elif txt == "Device":
                ttk.Combobox(
                    tr_frame,
                    textvariable=var,
                    values=["gpu", "cpu"],
                    state="readonly",
                ).grid(row=row, column=col_label + 1, sticky="ew", pady=4)
            elif txt == "Resume policy":
                ttk.Combobox(
                    tr_frame,
                    textvariable=var,
                    values=["auto", "never"],
                    state="readonly",
                ).grid(row=row, column=col_label + 1, sticky="ew", pady=4)
            else:
                entry = ttk.Entry(tr_frame, textvariable=var)
                entry.grid(row=row, column=col_label + 1, sticky="ew", pady=4)
                if txt in {"Epochs", "Image size (training)"}:
                    self._bind_numeric_validation(entry, allow_decimal=False)
                if txt == "Training label":
                    entry.insert(0, "training-name")
        gpu_cb = ttk.Checkbutton(tr_frame, text="Require GPU", variable=self.require_gpu_var)
        gpu_cb.grid(row=(len(training_fields) + 1) // 2, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))
        self._add_help_tooltip(gpu_cb, self._label_help("Require GPU"))
        self._update_base_checklist()

    # ------------------------------------------------------ tab Presets

    def _build_presets_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        dataset_preset_frame = ttk.LabelFrame(parent, text="Preset dataset", padding=10)
        dataset_preset_frame.pack(fill=tk.X, pady=(0, 10))
        dataset_preset_frame.columnconfigure(1, weight=1)
        dp_label = ttk.Label(dataset_preset_frame, text="Preset dataset")
        dp_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self._add_help_tooltip(dp_label, self._label_help("Preset dataset"))
        self.dataset_preset_combo_gen = ttk.Combobox(dataset_preset_frame, textvariable=self.dataset_preset_var, values=self.available_dataset_presets)
        self.dataset_preset_combo_gen.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(dataset_preset_frame, text="Applica preset dataset", command=self.apply_dataset_preset).grid(row=0, column=2, padx=(10, 0), pady=4)
        ttk.Label(dataset_preset_frame, text="Applica opzioni dataset preconfigurate senza toccare la lista immagini.", foreground="#555555").grid(row=1, column=0, columnspan=3, sticky=tk.W)

        training_preset_frame = ttk.LabelFrame(parent, text="Preset training", padding=10)
        training_preset_frame.pack(fill=tk.X, pady=(0, 10))
        training_preset_frame.columnconfigure(1, weight=1)
        tp_label = ttk.Label(training_preset_frame, text="Preset training")
        tp_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self._add_help_tooltip(tp_label, self._label_help("Preset training"))
        self.training_preset_combo_gen = ttk.Combobox(training_preset_frame, textvariable=self.training_preset_var, values=self.available_training_presets)
        self.training_preset_combo_gen.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(training_preset_frame, text="Applica preset training", command=self.apply_training_preset).grid(row=0, column=2, padx=(10, 0), pady=4)
        ttk.Label(training_preset_frame, text="Applica opzioni training preconfigurate.", foreground="#555555").grid(row=1, column=0, columnspan=3, sticky=tk.W)

        load_frame = ttk.LabelFrame(parent, text="Carica config completa (opzionale)", padding=10)
        load_frame.pack(fill=tk.X, pady=(0, 10))
        load_frame.columnconfigure(1, weight=1)
        cfg_label = ttk.Label(load_frame, text="Config completa")
        cfg_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self._add_help_tooltip(cfg_label, self._label_help("Config completa"))
        self.complete_config_combo = ttk.Combobox(load_frame, textvariable=self.complete_config_var, values=self.available_complete_configs)
        self.complete_config_combo.grid(row=0, column=1, sticky="ew", pady=4)
        btn_row = ttk.Frame(load_frame)
        btn_row.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        ttk.Button(btn_row, text="Carica config completa", command=self.load_complete_config).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Nuova da zero", command=self.load_blank).pack(side=tk.LEFT)
        ttk.Label(
            load_frame,
            text="Le config complete vengono cercate in configs/generated/ (default).",
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))

    # ------------------------------------------------------ tab Avanzate

    def _build_advanced_tab(self, parent: ttk.Frame) -> None:
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        dataset_adv_frame = ttk.LabelFrame(paned, text="Avanzate dataset + trasformazioni", padding=8)
        self._build_overrides_section(dataset_adv_frame, ["dataset_settings_overrides", "image_transform_overrides"])
        paned.add(dataset_adv_frame, weight=3)

        training_adv_frame = ttk.LabelFrame(paned, text="Avanzate training", padding=8)
        batch_row = ttk.Frame(training_adv_frame)
        batch_row.pack(fill=tk.X, pady=(0, 8))
        batch_label = ttk.Label(batch_row, text="Batch size")
        batch_label.pack(side=tk.LEFT, padx=(0, 8))
        self._add_help_tooltip(batch_label, self._label_help("Batch size"))
        batch_entry = ttk.Entry(batch_row, textvariable=self.batch_size_var, width=12)
        batch_entry.pack(side=tk.LEFT)
        self._bind_numeric_validation(batch_entry, allow_decimal=False)
        ttk.Label(batch_row, text="(default: 16)", foreground="#555555").pack(side=tk.LEFT, padx=(8, 0))
        self._build_overrides_section(training_adv_frame, ["training_overrides"])
        paned.add(training_adv_frame, weight=2)

    # ----------------------------------------- overrides section (shared)

    def _build_overrides_section(self, parent: ttk.Widget, section_names: list[str], *, max_height: int | None = None) -> None:
        ttk.Label(
            parent,
            text="Attiva solo le voci che vuoi cambiare. 'Preset/file' e' il valore caricato; 'Override' e' quello che verra' usato se attivi la voce.",
            wraplength=450,
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(0, 6))

        host = ttk.Frame(parent)
        host.pack(fill=tk.BOTH, expand=True)
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)

        canvas = tk.Canvas(host, height=(max_height or 200), highlightthickness=0)
        scrollbar = ttk.Scrollbar(host, orient=tk.VERTICAL, command=canvas.yview)
        content = ttk.Frame(canvas)
        content.bind("<Configure>", lambda _e, c=canvas: c.configure(scrollregion=c.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        ttk.Label(content, text="Usa", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky=tk.W, padx=(0, 4), pady=(0, 4))
        ttk.Label(content, text="Campo", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky=tk.W, padx=(0, 8), pady=(0, 4))
        ttk.Label(content, text="Preset / file", font=("Segoe UI", 9, "bold")).grid(row=0, column=2, sticky=tk.W, padx=(0, 8), pady=(0, 4))
        ttk.Label(content, text="Override", font=("Segoe UI", 9, "bold")).grid(row=0, column=3, sticky=tk.W, pady=(0, 4))

        row_idx = 1
        for section_name in section_names:
            specs = self.override_specs.get(section_name, [])
            if len(section_names) > 1:
                ttk.Label(content, text=section_name, font=("Segoe UI", 9, "bold"), foreground="#0b3a75").grid(
                    row=row_idx, column=0, columnspan=4, sticky=tk.W, pady=(6, 2)
                )
                row_idx += 1
            for field_name, default_value in specs:
                enabled_var = tk.BooleanVar(value=False)
                value_var = tk.StringVar(value=self._value_to_text(default_value))
                original_var = tk.StringVar(value=self._value_to_text(default_value))
                self.guided_override_enabled_vars[section_name][field_name] = enabled_var
                self.guided_override_value_vars[section_name][field_name] = value_var
                self.guided_original_value_vars[section_name][field_name] = original_var

                ttk.Checkbutton(content, variable=enabled_var).grid(row=row_idx, column=0, sticky=tk.W, padx=(0, 4), pady=2)
                lbl = ttk.Label(content, text=field_name)
                lbl.grid(row=row_idx, column=1, sticky=tk.W, padx=(0, 8), pady=2)
                self._add_help_tooltip(lbl, self._advanced_field_help(field_name, default_value))
                ttk.Entry(content, textvariable=original_var, state="readonly").grid(row=row_idx, column=2, sticky="ew", padx=(0, 8), pady=2)
                entry = ttk.Entry(content, textvariable=value_var)
                entry.grid(row=row_idx, column=3, sticky="ew", pady=2)
                entry.configure(state="disabled")
                enabled_var.trace_add("write", lambda *_a, e=entry, v=enabled_var: e.configure(state="normal" if v.get() else "disabled"))
                row_idx += 1

        content.columnconfigure(2, weight=1)
        content.columnconfigure(3, weight=1)

    def _iter_uppercase_settings(self, cls: type, *, exclude: set[str] | None = None) -> list[tuple[str, object]]:
        excluded = exclude or set()
        items: list[tuple[str, object]] = []
        for attr_name in dir(cls):
            if not attr_name.isupper() or attr_name in excluded:
                continue
            items.append((attr_name.lower(), getattr(cls, attr_name)))
        return items

    def _collect_override_specs(self) -> dict[str, list[tuple[str, object]]]:
        dataset_exclude = {
            "DATASET_ROOT",
            "FIRE_IMAGE_PATHS",
            "NUM_IMAGES",
            "IMAGE_SIZE",
            "NEGATIVE_RATIO",
            "TRAIN_SPLIT",
            "DEMO_MODE",
            "DEMO_WAIT_MS",
        }
        training_exclude = {
            "MODEL_SIZE",
            "DEVICE",
            "EPOCHS",
            "BATCH_SIZE",
            "IMAGE_SIZE",
            "PROJECT_NAME",
            "EXPERIMENT_NAME",
            "OVERWRITE_EXISTING",
            "VERBOSE",
        }

        return {
            "dataset_settings_overrides": self._iter_uppercase_settings(DatasetGenerationSettings, exclude=dataset_exclude),
            "image_transform_overrides": self._iter_uppercase_settings(ImageTransformSettings),
            "training_overrides": self._iter_uppercase_settings(TrainingSettings, exclude=training_exclude),
        }

    def _value_to_text(self, value: object) -> str:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return str(value)
        return yaml.safe_dump(value, default_flow_style=True, sort_keys=False, allow_unicode=False).strip()

    def _parse_dynamic_value(self, raw_value: str, field_name: str) -> object:
        text = raw_value.strip()
        if not text:
            raise ValueError(f"Valore mancante per {field_name}")
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Valore non valido per {field_name}: {exc}") from exc

    def _sync_guided_from_config(self, config: dict) -> None:
        """Update preset/file-origin values without altering session overrides."""
        section_payloads = {
            "dataset_settings_overrides": config.get("dataset_settings_overrides", {}),
            "image_transform_overrides": config.get("image_transform_overrides", {}),
            "training_overrides": config.get("training_overrides", {}),
        }
        for section_name, payload in section_payloads.items():
            data = payload if isinstance(payload, dict) else {}
            for field_name, original_var in self.guided_original_value_vars[section_name].items():
                if field_name in data:
                    original_var.set(self._value_to_text(data[field_name]))
                else:
                    original_var.set("(non impostato)")

    def _collect_guided_overrides(self, section_name: str) -> dict:
        payload: dict[str, object] = {}
        for field_name, enabled_var in self.guided_override_enabled_vars[section_name].items():
            if not enabled_var.get():
                continue
            raw_value = self.guided_override_value_vars[section_name][field_name].get()
            payload[field_name] = self._parse_dynamic_value(raw_value, f"{section_name}.{field_name}")
        return payload

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
        self._update_base_checklist()

    def remove_selected_fire_image_paths(self) -> None:
        for index in reversed(self.fire_image_paths_listbox.curselection()):
            self.fire_image_paths_listbox.delete(index)
        self._update_base_checklist()

    def clear_fire_image_paths(self) -> None:
        self.fire_image_paths_listbox.delete(0, tk.END)
        self._update_base_checklist()

    def _apply_config_to_form(self, config: dict) -> None:
        project = config.get("project", {})
        dataset = config.get("dataset", {})
        training = config.get("training", {})
        full_root = str(project.get("persistent_root", ""))
        if full_root.startswith(CLOUD_PERSISTENT_ROOT_PREFIX):
            self.target_env_var.set("cloud")
        elif full_root.startswith(LOCAL_PERSISTENT_ROOT_PREFIX):
            self.target_env_var.set("local")

        self.dataset_label_var.set(str(dataset.get("label", "")))
        self.num_images_var.set(self._to_form_text(dataset.get("num_images", "")))
        self.image_size_var.set(self._to_form_text(dataset.get("image_size", "")))
        self.negative_ratio_var.set(self._to_form_text(dataset.get("negative_ratio", "")))
        self.train_split_var.set(self._to_form_text(dataset.get("train_split", "")))
        self.dataset_seed_var.set(self._to_form_text(dataset.get("seed", "")))
        self.force_regenerate_var.set(bool(dataset.get("force_regenerate", False)))

        self.training_label_var.set(str(training.get("label", "")))
        self.model_size_var.set(self._to_model_size_display(str(training.get("model_size", ""))))
        self.device_var.set(self._to_device_display(str(training.get("device", ""))))
        self.require_gpu_var.set(bool(training.get("require_gpu", True)))
        self.require_gpu_var.set(bool(training.get("require_gpu", True)))
        self.epochs_var.set(self._to_form_text(training.get("epochs", "")))
        self.batch_size_var.set(self._to_form_text(training.get("batch_size", "")))
        self.training_image_size_var.set(self._to_form_text(training.get("image_size", "")))
        self.resume_policy_var.set(str(training.get("resume", "")))

        fire_paths = [str(item) for item in dataset.get("fire_image_paths", [])]
        self._set_listbox_items(self.fire_image_paths_listbox, fire_paths)
        self._sync_guided_from_config(config)
        self._update_project_label_preview()
        self._update_base_checklist()

    def _parse_int(self, value: str, label: str) -> int:
        cleaned = value.strip()
        try:
            if not cleaned:
                raise ValueError("empty")
            if "." in cleaned:
                parsed = float(cleaned)
                if not parsed.is_integer():
                    raise ValueError("not integer")
                return int(parsed)
            return int(cleaned)
        except ValueError as exc:
            raise ValueError(f"{label} deve essere un intero") from exc

    def _parse_float(self, value: str, label: str) -> float:
        cleaned = value.strip()
        try:
            if not cleaned:
                raise ValueError("empty")
            return float(cleaned)
        except ValueError as exc:
            raise ValueError(f"{label} deve essere un numero") from exc

    def _to_form_text(self, value: object) -> str:
        if value is None:
            return ""
        return str(value)

    def build_config(self) -> dict:
        config = deepcopy(self.loaded_config)
        config.setdefault("project", {})
        config.setdefault("dataset", {})
        config.setdefault("training", {})

        fire_paths = self._get_listbox_items(self.fire_image_paths_listbox)
        if not fire_paths:
            raise ValueError("Serve almeno un'immagine di fuoco di base nella tab Generale")

        config["project"]["label"] = self.project_label_preview_var.get().strip() or "fire-project"
        env = self.target_env_var.get().strip().lower()
        if env == "cloud":
            config["project"]["persistent_root"] = "artifacts/cloud"
        else:
            config["project"]["persistent_root"] = "artifacts/local"
        config["dataset"]["label"] = self.dataset_label_var.get().strip()
        config["dataset"]["fire_image_paths"] = fire_paths
        config["dataset"]["num_images"] = self._parse_int(self.num_images_var.get(), "num_images")
        config["dataset"]["image_size"] = self._parse_int(self.image_size_var.get(), "image_size (dataset)")
        config["dataset"]["negative_ratio"] = self._parse_float(self.negative_ratio_var.get(), "negative_ratio")
        config["dataset"]["train_split"] = self._parse_float(self.train_split_var.get(), "train_split")
        config["dataset"]["seed"] = self._parse_int(self.dataset_seed_var.get(), "dataset_seed")
        config["dataset"]["force_regenerate"] = bool(self.force_regenerate_var.get())

        config["training"]["label"] = self.training_label_var.get().strip()
        config["training"]["model_size"] = self._to_model_size_code(self.model_size_var.get())
        config["training"].pop("weights", None)
        config["training"]["device"] = self._to_device_config(self.device_var.get())
        config["training"]["require_gpu"] = bool(self.require_gpu_var.get())
        config["training"]["epochs"] = self._parse_int(self.epochs_var.get(), "epochs")
        batch_raw = self.batch_size_var.get().strip()
        config["training"]["batch_size"] = self._parse_int(batch_raw, "batch_size") if batch_raw else 16
        config["training"]["image_size"] = self._parse_int(self.training_image_size_var.get(), "image_size (training)")
        config["training"]["resume"] = self.resume_policy_var.get().strip()

        config["dataset_settings_overrides"] = self._collect_guided_overrides("dataset_settings_overrides")
        config["image_transform_overrides"] = self._collect_guided_overrides("image_transform_overrides")
        config["training_overrides"] = self._collect_guided_overrides("training_overrides")
        config.pop("extends", None)
        return config

    def save_config(self) -> None:
        try:
            config = self.build_config()
            requested_name = Path(self.save_name_var.get().strip()).name
            if not requested_name:
                requested_name = "config.yaml"
            if not requested_name.endswith(".yaml"):
                requested_name += ".yaml"

            target_relative = Path(GENERATED_DIR_NAME) / requested_name
            target_path = CONFIGS_DIR / target_relative
            latest_relative, latest_meta_relative, latest_kind = self._resolve_latest_targets(self.target_env_var.get())
            latest_path = CONFIGS_DIR / latest_relative
            latest_meta_path = CONFIGS_DIR / latest_meta_relative

            write_yaml(target_path, config)
            write_yaml(latest_path, config)
            write_yaml(
                latest_meta_path,
                {
                    "schema_version": 2,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "complete_config": self.complete_config_var.get().strip() or None,
                    "dataset_preset": self.dataset_preset_var.get().strip() or None,
                    "training_preset": self.training_preset_var.get().strip() or None,
                    "target_environment": self.target_env_var.get().strip().lower(),
                    "saved_config": target_relative.as_posix(),
                    "latest_config": latest_relative.as_posix(),
                    "latest_kind": latest_kind,
                    "app_version": APP_VERSION,
                },
            )

            self.refresh_options()
            messagebox.showinfo(
                "Config salvata",
                (
                    f"Config scritta in:\n- {target_path}\n- {latest_path}\n\n"
                    f"Metadata latest {latest_kind}:\n- {latest_meta_path}"
                ),
            )
        except Exception as exc:
            messagebox.showerror("Errore salvataggio", str(exc))


def main() -> None:
    app = ExperimentConfiguratorApp()
    app.mainloop()


if __name__ == "__main__":
    main()