"""
Fire Detection script using YOLOv8

Detects fire in real-time from multiple sources:
1. Webcam
2. RTMP/RTSP streams
3. Video files
4. Static images (for testing/validation)

Usage:
    # Detection from webcam (default)
    python detect.py
    
    # Detection from RTMP stream
    python detect.py --source rtmp://server/app/stream
    
    # Detection from video file
    python detect.py --source video.mp4
    
    # Test on static images (validation set)
    python detect.py --source dataset/images/val/
    
    # With custom model and confidence
    python detect.py --source dataset/images/val/ --weights best.pt --conf 0.6

Controls:
    - Press 'q' or 'ESC' to quit
    - Press 's' to save current frame with detections
    - For image testing: Use arrow keys (or 'a'/'d') to navigate between images
"""

from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import importlib
import os
from pathlib import Path
from datetime import datetime
import threading
import time

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_portable_artifact_path(path_value: str, pointer_path: Path) -> str:
    """Resolve a relative artifact path from the persistent root or keep absolute paths."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)

    sibling_root = pointer_path.parent
    sibling_candidate = (sibling_root / candidate).resolve()
    if sibling_candidate.exists():
        return str(sibling_candidate)

    persistent_root = pointer_path.parent.parent
    return str((persistent_root / candidate).resolve())


def load_pointer_payload(path: Path) -> dict | None:
    """Load YAML metadata payloads."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return None

    return payload if isinstance(payload, dict) else None


def resolve_default_model_path() -> str:
    """Resolve the preferred trained model path from the persistent registry."""
    candidate_files = [
        PROJECT_ROOT / "artifacts/local/exports/latest.yaml",
        PROJECT_ROOT / "artifacts/cloud-notebook-local/exports/latest.yaml",
    ]

    for latest_path in sorted(PROJECT_ROOT.glob("artifacts/**/exports/latest.yaml")):
        if latest_path not in candidate_files:
            candidate_files.append(latest_path)

    for latest_path in sorted(PROJECT_ROOT.glob("artifacts/**/latest.yaml")):
        if latest_path not in candidate_files:
            candidate_files.append(latest_path)

    for candidate in candidate_files:
        if candidate.suffix.lower() in {".yaml", ".yml"} and candidate.exists():
            payload = load_pointer_payload(candidate)
            if payload is None:
                continue

            model_path = payload.get("model_path")
            if isinstance(model_path, str):
                resolved_model_path = resolve_portable_artifact_path(model_path, candidate)
                if os.path.exists(resolved_model_path):
                    return resolved_model_path

    return str(PROJECT_ROOT / "artifacts/local/exports/latest.yaml")


class FireDetector:
    """Fire detection system using YOLOv8"""
    
    # Variabili globali per la cattura dei tasti
    _last_key = None
    _key_listener = None
    
    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.5,
        device: str = "cpu",
    ):
        """
        Inizializza il detector.
        
        Args:
            model_path: Percorso del modello YOLOv8
            conf_threshold: Soglia di confidenza per le detections
            device: Device per inference ('cpu' o numero GPU come stringa)
        """
        normalized_model_path = (model_path or "").strip()
        if normalized_model_path.lower() in {"", "latest", "auto", "default"}:
            resolved_model_path = resolve_default_model_path()
        else:
            resolved_model_path = normalized_model_path

        if not os.path.exists(resolved_model_path):
            raise FileNotFoundError(
                f"Modello non trovato: {resolved_model_path}\n"
                "Esegui prima: python run_experiment.py --config configs/generated/latest.local.yaml"
            )
        
        print(f"Caricamento modello: {resolved_model_path}")
        self.model = YOLO(resolved_model_path)
        self.model_name = Path(resolved_model_path).name
        self.conf_threshold = conf_threshold
        self.device = device
        self._highgui_available = True
        self._highgui_warning_printed = False
        
        # Crea cartella per i salvataggi
        self.save_dir = Path("detections")
        self.save_dir.mkdir(exist_ok=True)
        
        print(f"✓ Modello caricato con successo")
        print(f"  Soglia di confidenza: {conf_threshold}")
        print(f"  Device: {'GPU' if device != 'cpu' else 'CPU'}")

    def _disable_highgui(self, reason: str) -> None:
        """Disable OpenCV GUI features after a runtime error."""
        self._highgui_available = False
        if not self._highgui_warning_printed:
            print("\n⚠️ OpenCV GUI non disponibile in questo ambiente.")
            print(f"   Dettaglio: {reason}")
            print("   Continuo in modalita' headless (nessuna finestra).")
            print("   Premi Ctrl+C nel terminale per interrompere.")
            self._highgui_warning_printed = True

    def _imshow_safe(self, window_name: str, frame: np.ndarray) -> bool:
        """Show a frame when HighGUI is available, otherwise switch to headless mode."""
        if not self._highgui_available:
            return False
        try:
            cv2.imshow(window_name, frame)
            return True
        except cv2.error as ex:
            self._disable_highgui(str(ex))
            return False

    def _wait_key_safe(self, timeout_ms: int = 1) -> int:
        """Read keyboard input through OpenCV only when HighGUI is available."""
        if not self._highgui_available:
            return -1
        try:
            return cv2.waitKey(timeout_ms)
        except cv2.error as ex:
            self._disable_highgui(str(ex))
            return -1

    def _destroy_windows_safe(self) -> None:
        """Close OpenCV windows safely in GUI or headless environments."""
        if not self._highgui_available:
            return
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    
    @staticmethod
    def _on_key_press(key):
        """Callback per la pressione dei tasti (con pynput)."""
        try:
            FireDetector._last_key = key.char
        except AttributeError:
            # Tasto speciale (freccia, etc)
            key_name = str(key).split('.')[-1]
            if 'left' in key_name:
                FireDetector._last_key = 'left'
            elif 'right' in key_name:
                FireDetector._last_key = 'right'
    
    @staticmethod
    def _start_key_listener():
        """Avvia il listener dei tasti usando pynput (se disponibile)."""
        try:
            keyboard = importlib.import_module("pynput.keyboard")
            listener = keyboard.Listener(on_press=FireDetector._on_key_press)
            listener.start()
            FireDetector._key_listener = listener
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _get_key_robust(timeout_ms=0):
        """
        Legge un tasto in modo robusto.
        Prima tenta con pynput (System-wide), poi fallback a OpenCV.
        """
        # Se abbiamo un tasto in sospeso da pynput, ritornalo
        if FireDetector._last_key is not None:
            key = FireDetector._last_key
            FireDetector._last_key = None
            return key
        
        # Altrimenti usa OpenCV
        try:
            key = cv2.waitKey(timeout_ms if timeout_ms > 0 else 0)
        except cv2.error:
            key = -1
        return key
    
    def detect_frame(self, frame: np.ndarray) -> tuple:
        """
        Esegue detection su un singolo frame.
        
        Args:
            frame: Frame (immagine) da analizzare
        
        Returns:
            Tuple[np.ndarray, List]: (frame annotato, list di detections)
        """
        # Esegui detection
        results = self.model(
            frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )
        
        # Annota il frame
        annotated_frame = results[0].plot()
        
        # Estrai detections
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detection = {
                    'class_id': int(box.cls[0]),
                    'class_name': 'FIRE',
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].cpu().numpy(),
                }
                detections.append(detection)
        
        return annotated_frame, detections
    
    def draw_info(
        self,
        frame: np.ndarray,
        detections: list,
        fps: float = 0,
    ) -> np.ndarray:
        """
        Aggiunge informazioni al frame (FPS, numero fires, etc).
        
        Args:
            frame: Frame da annotare
            detections: Lista di detections
            fps: Frame per secondo
        
        Returns:
            np.ndarray: Frame annotato
        """
        # FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Numero di fires detected
        num_fires = len(detections)
        fire_color = (0, 0, 255) if num_fires > 0 else (0, 255, 0)
        cv2.putText(
            frame,
            f"FIRES: {num_fires}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            fire_color,
            2
        )
        
        # Confidence scores
        if num_fires > 0:
            y_offset = 110
            for i, det in enumerate(detections):
                text = f"Fire {i+1}: {det['confidence']:.2%}"
                cv2.putText(
                    frame,
                    text,
                    (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    1
                )

        # Modello attivo (in piccolo, in basso a destra)
        model_text = f"model: {self.model_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(model_text, font, font_scale, thickness)
        x = max(10, frame.shape[1] - text_w - 10)
        y = max(text_h + 10, frame.shape[0] - 10)

        cv2.rectangle(
            frame,
            (x - 4, y - text_h - 4),
            (x + text_w + 4, y + baseline + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            model_text,
            (x, y),
            font,
            font_scale,
            (220, 220, 220),
            thickness,
            cv2.LINE_AA,
        )
        
        return frame
    
    def save_frame(self, frame: np.ndarray, detections: list) -> str:
        """
        Salva il frame con le detections.
        
        Args:
            frame: Frame da salvare
            detections: Lista di detections
        
        Returns:
            str: Percorso del file salvato
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fire_detection_{timestamp}_{len(detections)}fires.jpg"
        filepath = self.save_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"✓ Frame salvato: {filepath}")
        return str(filepath)

    def _open_camera_capture(self, camera_id: int) -> cv2.VideoCapture:
        """Open and preconfigure a camera capture."""
        if os.name == "nt":
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _probe_camera_source(self, camera_id: int) -> dict:
        """Probe a camera source and collect basic availability diagnostics."""
        cap = self._open_camera_capture(camera_id)
        info = {
            "camera_id": camera_id,
            "available": False,
            "resolution": None,
            "message": "Non disponibile",
            "non_black": False,
        }

        if not cap.isOpened():
            cap.release()
            return info

        frame = None
        for _ in range(6):
            ret, candidate = cap.read()
            if ret and candidate is not None:
                frame = candidate
                break
            time.sleep(0.15)

        if frame is None:
            cap.release()
            info["message"] = "Nessun frame"
            return info

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        info["non_black"] = bool(frame.max() > 20 and frame.mean() > 5)
        cap.release()

        info["available"] = True
        info["resolution"] = f"{width}x{height}"
        info["message"] = "Frame valido" if info["non_black"] else "Frame molto scuro/nero"
        return info

    def select_camera_source(self, max_sources: int = 6) -> int:
        """List available cameras in the terminal and prompt the user to choose one."""

        def scan_sources() -> list[dict]:
            sources: list[dict] = []
            consecutive_missing = 0
            found_any = False
            for camera_id in range(max_sources):
                source = self._probe_camera_source(camera_id)
                sources.append(source)
                if source["available"]:
                    found_any = True
                    consecutive_missing = 0
                else:
                    consecutive_missing += 1
                    if found_any and consecutive_missing >= 2:
                        break
            return sources

        sources = scan_sources()
        available_sources = [source for source in sources if source["available"]]
        if not available_sources:
            raise RuntimeError("Nessuna camera disponibile trovata")

        print("\n" + "=" * 60)
        print("SELEZIONE CAMERA")
        print("=" * 60)
        print("Le camere con '*' hanno restituito un frame non nero.")
        print()

        for source in available_sources:
            marker = "*" if source["non_black"] else " "
            resolution = source["resolution"] or "?x?"
            print(f"[{source['camera_id']}] {marker} Camera {source['camera_id']}  {resolution}  |  {source['message']}")

        preferred_source = next(
            (source for source in available_sources if source["non_black"]),
            available_sources[0],
        )

        while True:
            choice = input(
                f"\nSeleziona camera ID [{preferred_source['camera_id']}] oppure 'r' per rescansionare, 'q' per annullare: "
            ).strip().lower()

            if not choice:
                return int(preferred_source["camera_id"])

            if choice == "q":
                raise RuntimeError("Selezione camera annullata")

            if choice == "r":
                sources = scan_sources()
                available_sources = [source for source in sources if source["available"]]
                if not available_sources:
                    raise RuntimeError("Nessuna camera disponibile trovata")
                print()
                for source in available_sources:
                    marker = "*" if source["non_black"] else " "
                    resolution = source["resolution"] or "?x?"
                    print(f"[{source['camera_id']}] {marker} Camera {source['camera_id']}  {resolution}  |  {source['message']}")
                preferred_source = next(
                    (source for source in available_sources if source["non_black"]),
                    available_sources[0],
                )
                continue

            if choice.isdigit():
                selected = int(choice)
                if any(source["camera_id"] == selected and source["available"] for source in available_sources):
                    return selected

            print("Scelta non valida. Inserisci un ID disponibile, 'r' o 'q'.")
    
    def run_webcam(self, camera_id: int | None = None) -> None:
        """
        Esegue la detection da webcam.
        
        Args:
            camera_id: ID della webcam. Se omesso apre un selettore testuale.
        """
        prompt_for_next_camera = camera_id is None

        while True:
            selected_camera_id = camera_id if camera_id is not None else self.select_camera_source()

            print(f"\n{'='*60}")
            print(f"🎥 APERTURA WEBCAM {selected_camera_id}")
            print(f"{'='*60}")

            cap = self._open_camera_capture(selected_camera_id)

            if not cap.isOpened():
                print(f"❌ Errore: impossibile aprire la webcam {selected_camera_id}")
                print(f"\n🔧 Troubleshooting:")
                print(f"  - Verifica che la webcam sia collegata")
                print(f"  - Prova con un altro ID: python detect.py --source 1")
                print(f"  - Su Linux potrebbe essere /dev/video0 invece di 0")
                print(f"  - Controlla permessi della webcam")
                raise RuntimeError(f"Impossibile aprire webcam {selected_camera_id}")

            print(f"✓ Webcam {selected_camera_id} aperta")
            print(f"⚙️ Configurazione webcam...")

            print(f"⏳ Inizializzazione camera (5 tentativi)...")
            frame = None
            for i in range(5):
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"  Tentativo {i+1}: ✓ Frame ricevuto")
                    frame = test_frame
                    break
                print(f"  Tentativo {i+1}: ✗ Nessun frame")
                time.sleep(0.3)

            if frame is None:
                print(f"❌ Webcam non ha risposto dopo 5 tentativi")
                print(f"\n🔧 Troubleshooting:")
                print(f"  - La webcam potrebbe essere in uso da un'altra app")
                print(f"  - Prova a riavviare la webcam")
                print(f"  - Prova a riavviare il sistema")
                cap.release()
                raise RuntimeError(f"Webcam {selected_camera_id} non risponde")

            print(f"✓ Webcam inizializzata correttamente")
            print(f"\n📊 Proprietà webcam:")
            print(f"  Risoluzione: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")
            print(f"  Frame shape: {frame.shape}")
            print(f"  Frame dtype: {frame.dtype}")
            print(f"  Min pixel value: {frame.min()}, Max: {frame.max()}")

            if frame.max() < 10:
                print(f"\n⚠️ ATTENZIONE: Frame sembra essere nero/vuoto!")
                print(f"   Pixel values are all very low (max={frame.max()})")
                print(f"   Prova a:")
                print(f"   - Attendi 3-5 secondi per la stabilizzazione camera")
                print(f"   - Verifica che la webcam abbia illuminazione adeguata")
                print(f"   - Prova con una telecamera diversa")

            print(f"\n{'='*60}")
            print(f"DETECTION DA WEBCAM AVVIATA")
            print(f"{'='*60}")
            print(f"Premi 'q' o 'ESC' per uscire")
            print(f"Premi 's' per salvare il frame")
            if prompt_for_next_camera:
                print(f"Alla chiusura tornerai al selettore camera")
            print(f"{'='*60}\n")

            fps_time = time.time()
            fps_counter = 0
            fps = 0
            frame_count = 0

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("❌ Errore: impossibile leggere il frame")
                        break

                    frame_count += 1

                    if frame_count % 30 == 0 and frame.max() < 10:
                        print(f"⚠️ Frame {frame_count}: Ancora nero (max pixel={frame.max()})")

                    frame = cv2.flip(frame, 1)
                    annotated_frame, detections = self.detect_frame(frame)
                    annotated_frame = self.draw_info(annotated_frame, detections, fps)
                    self._imshow_safe("Fire Detection - Webcam", annotated_frame)

                    fps_counter += 1
                    if time.time() - fps_time > 1:
                        fps = fps_counter / (time.time() - fps_time)
                        fps_counter = 0
                        fps_time = time.time()

                    key = self._wait_key_safe(1) & 0xFF
                    if key in [ord('q'), 27]:
                        break
                    if key == ord('s'):
                        self.save_frame(annotated_frame, detections)

            finally:
                cap.release()
                self._destroy_windows_safe()
                print(f"\n✓ Detection da webcam terminata ({frame_count} frame elaborati)")

            if not prompt_for_next_camera:
                break

            camera_id = None
    
    def run_rtmp(self, rtmp_url: str) -> None:
        """
        Esegue la detection da stream RTMP.
        
        Args:
            rtmp_url: URL dello stream RTMP
        """
        cap = cv2.VideoCapture(rtmp_url)
        
        if not cap.isOpened():
            raise RuntimeError(f"Errore: impossibile connettersi a {rtmp_url}")
        
        print("\n" + "="*60)
        print("DETECTION DA RTMP STREAM")
        print("="*60)
        print(f"URL: {rtmp_url}")
        print("Premi 'q' o 'ESC' per uscire")
        print("Premi 's' per salvare il frame")
        print("="*60 + "\n")
        
        import time
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        dropped_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("ATTENZIONE: frame non disponibile (possibile disconnessione)")
                    dropped_frames += 1
                    if dropped_frames > 30:  # Se perdiamo 30 frame di fila, riconnetti
                        print("Tentativo di riconnessione...")
                        cap = cv2.VideoCapture(rtmp_url)
                        dropped_frames = 0
                    continue
                
                dropped_frames = 0
                
                # Riduci la risoluzione se necessario per performanza
                height = frame.shape[0]
                if height > 720:
                    scale = 720 / height
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                
                # Esegui detection
                annotated_frame, detections = self.detect_frame(frame)
                
                # Aggiungi info
                annotated_frame = self.draw_info(annotated_frame, detections, fps)
                
                # Mostra il frame
                self._imshow_safe("Fire Detection - RTMP Stream", annotated_frame)
                
                # Calcola FPS
                fps_counter += 1
                if time.time() - fps_time > 1:
                    fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()
                
                # Gestisci input
                key = self._wait_key_safe(1) & 0xFF
                if key in [ord('q'), 27]:  # 'q' o ESC
                    break
                elif key == ord('s'):
                    self.save_frame(annotated_frame, detections)
        
        except KeyboardInterrupt:
            print("\n\nInterrotto dall'utente")
        finally:
            cap.release()
            self._destroy_windows_safe()
            print("\n✓ Detection da RTMP terminata")
    
    def run_video_file(self, video_path: str) -> None:
        """
        Esegue la detection da file video.
        
        Args:
            video_path: Percorso del file video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File non trovato: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Errore: impossibile aprire il video {video_path}")
        
        # Estrai informazioni dal video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        
        print("\n" + "="*60)
        print("DETECTION DA FILE VIDEO")
        print("="*60)
        print(f"File: {video_path}")
        print(f"Frame totali: {total_frames}")
        print(f"FPS originale: {fps_video:.2f}")
        print("Premi 'q' o 'ESC' per uscire")
        print("Premi 's' per salvare il frame")
        print("="*60 + "\n")
        
        import time
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                
                # Esegui detection
                annotated_frame, detections = self.detect_frame(frame)
                
                # Aggiungi info
                annotated_frame = self.draw_info(annotated_frame, detections, fps)
                
                # Aggiungi numero frame
                progress_text = f"Frame: {frame_num}/{total_frames}"
                cv2.putText(
                    annotated_frame,
                    progress_text,
                    (10, annotated_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
                # Mostra il frame
                self._imshow_safe("Fire Detection - Video File", annotated_frame)
                
                # Calcola FPS
                fps_counter += 1
                if time.time() - fps_time > 1:
                    fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()
                
                # Gestisci input
                key = self._wait_key_safe(1) & 0xFF
                if key in [ord('q'), 27]:  # 'q' o ESC
                    break
                elif key == ord('s'):
                    self.save_frame(annotated_frame, detections)
        
        finally:
            cap.release()
            self._destroy_windows_safe()
            print(f"\n✓ Detection completata: {frame_num} frame processati")


    def test_on_images(self, images_folder: str) -> None:
        """
        Testa il modello su immagini statiche da una cartella.
        
        Args:
            images_folder: Percorso della cartella contenente le immagini
        """
        if not os.path.exists(images_folder):
            raise FileNotFoundError(f"Cartella non trovata: {images_folder}")
        
        # Trova tutte le immagini
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        for file in os.listdir(images_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(images_folder, file))
        
        if not image_paths:
            print(f"Nessuna immagine trovata nella cartella: {images_folder}")
            return
        
        image_paths.sort()  # Ordina alfabeticamente

        if not self._highgui_available:
            raise RuntimeError("OpenCV GUI non disponibile: test_on_images richiede finestre interattive")
        
        print("\n" + "="*60)
        print("TEST SU IMMAGINI STATICHE")
        print("="*60)
        print(f"Cartella: {images_folder}")
        print(f"Immagini trovate: {len(image_paths)}")
        print("Controlli:")
        print("  ← → (frecce) o 'a'/'d': Naviga tra le immagini")
        print("  's' : Salva immagine corrente")
        print("  'q' o ESC: Esci")
        print("="*60 + "\n")
        
        # Inizia il listener dei tasti (se disponibile)
        use_pynput = self._start_key_listener()
        if use_pynput:
            print("✅ Usando pynput per la cattura dei tasti (system-wide)")
        else:
            print("ℹ️  Usando OpenCV per la cattura dei tasti (window-focused)")
        print()
        
        current_idx = 0
        
        while True:
            # Carica immagine corrente
            img_path = image_paths[current_idx]
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Errore nel caricamento: {img_path}")
                current_idx = (current_idx + 1) % len(image_paths)
                continue
            
            # Ridimensiona se troppo grande per il display
            height, width = img.shape[:2]
            max_display_size = 1200
            if max(height, width) > max_display_size:
                scale = max_display_size / max(height, width)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            # Esegui detection
            annotated_img, detections = self.detect_frame(img)
            
            # Aggiungi info
            annotated_img = self.draw_info(annotated_img, detections)
            
            # Aggiungi nome file e contatore
            filename = os.path.basename(img_path)
            info_text = f"{filename} ({current_idx + 1}/{len(image_paths)})"
            cv2.putText(
                annotated_img,
                info_text,
                (10, annotated_img.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Mostra immagine
            if not self._imshow_safe("Fire Detection - Test Images", annotated_img):
                raise RuntimeError("OpenCV GUI non disponibile: impossibile mostrare immagini interattive")
            
            # Gestisci input in modo robusto
            navigate = True
            while navigate:
                key = self._get_key_robust()
                
                # Converti key a stringa se è un numero
                if isinstance(key, int):
                    key_char = chr(key & 0xFF) if key >= 0 else None
                else:
                    key_char = str(key).lower() if key else None
                
                # Gestione dei tasti
                if key == 27 or key_char == 'q':  # ESC o q
                    navigate = False
                    break
                elif key_char == 's':  # Salva
                    self.save_frame(annotated_img, detections)
                    navigate = False
                elif key_char in ['a', 'left'] or key in [81, 65361]:  # Sinistra
                    current_idx = (current_idx - 1) % len(image_paths)
                    navigate = False
                elif key_char in ['d', 'right'] or key in [83, 65363]:  # Destra
                    current_idx = (current_idx + 1) % len(image_paths)
                    navigate = False
                elif key_char and key_char not in ['\x00', '']:
                    # Tasto riconosciuto ma non valido, mostra aiuto ma continua
                    continue
                else:
                    # Timeout, continua
                    continue
            
            # Ferma il listener se eravamo fuori dal loop
            if key == 27 or key_char == 'q':
                break
        
        self._destroy_windows_safe()
        
        # Ferma il listener dei tasti
        if self._key_listener is not None:
            self._key_listener.stop()
        
        print("\n✓ Test su immagini completato")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Fire Detection using YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python detect.py                              # Selettore camera testuale (default)
    python detect.py --source webcam              # Selettore camera testuale
  python detect.py --source 0                   # Webcam con ID 0
  python detect.py --source rtmp://example.com/live/stream   # RTMP stream
  python detect.py --source video.mp4           # File video
  python detect.py --source dataset/images/val/ # Test su immagini validation
    python detect.py --weights best.pt            # Modello personalizzato
  python detect.py --conf 0.6                   # Soglia confidenza 0.6
        """
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help="Sorgente: webcam/camera=selettore testuale, numero=webcam diretta, RTMP/RTSP URL=stream, path file=video, path cartella=immagini"
    )
    parser.add_argument(
        "--weights",
        type=str,
        nargs="?",
        const="latest",
        default="latest",
        help="Percorso del modello YOLOv8. Se omesso (o latest/auto/default) usa il modello registrato piu' recente."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Soglia di confidenza (0-1, default: 0.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device per inference: 'cpu' o numero GPU (default: cpu)"
    )
    
    args = parser.parse_args()

    try:
        # Inizializza il detector
        detector = FireDetector(
            model_path=args.weights,
            conf_threshold=args.conf,
            device=args.device,
        )
        
        # Determina la sorgente
        source = args.source

        if source.lower() in {"webcam", "camera", "select"}:
            detector.run_webcam(camera_id=None)
        
        
        # Se è un numero, è la webcam
        elif source.isdigit():
            camera_id = int(source)
            detector.run_webcam(camera_id=camera_id)
        
        # Se inizia con rtmp:// o rtsp://, è uno stream
        elif source.startswith(("rtmp://", "rtsp://")):
            detector.run_rtmp(rtmp_url=source)
        
        # Se è una cartella esistente, testa su immagini statiche
        elif os.path.isdir(source):
            detector.test_on_images(images_folder=source)
        
        # Se è un file esistente, è un video
        elif os.path.isfile(source):
            detector.run_video_file(video_path=source)
        
        # Altrimenti errore
        else:
            print(f"❌ Sorgente non riconosciuta: {source}")
            print("Tipi supportati:")
            print("  - Numero (es: 0, 1): Webcam")
            print("  - URL RTMP/RTSP: Stream video")
            print("  - Path cartella: Immagini statiche")
            print("  - Path file: Video locale")
            return
    
    except FileNotFoundError as e:
        print(f"\n❌ Errore: {e}")
        return
    except RuntimeError as e:
        if str(e) == "Selezione camera annullata":
            print("\n✓ Selezione camera annullata, uscita.")
            return
        print(f"\n❌ Errore: {e}")
        return
    except KeyboardInterrupt:
        print("\n\n✓ Interrotto dall'utente")


if __name__ == "__main__":
    main()
