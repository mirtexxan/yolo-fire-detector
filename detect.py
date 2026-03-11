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
import os
from pathlib import Path
from datetime import datetime
import threading
import time


class FireDetector:
    """Fire detection system using YOLOv8"""
    
    # Variabili globali per la cattura dei tasti
    _last_key = None
    _key_listener = None
    
    def __init__(
        self,
        model_path: str = "runs/detect/fire_detector_runs/train/weights/best.pt",
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
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modello non trovato: {model_path}\n"
                f"Esegui prima: python train.py"
            )
        
        print(f"Caricamento modello: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Crea cartella per i salvataggi
        self.save_dir = Path("detections")
        self.save_dir.mkdir(exist_ok=True)
        
        print(f"✓ Modello caricato con successo")
        print(f"  Soglia di confidenza: {conf_threshold}")
        print(f"  Device: {'GPU' if device != 'cpu' else 'CPU'}")
    
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
            from pynput import keyboard
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
        key = cv2.waitKey(timeout_ms if timeout_ms > 0 else 0)
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
    
    def run_webcam(self, camera_id: int = 0) -> None:
        """
        Esegue la detection da webcam.
        
        Args:
            camera_id: ID della webcam (default: 0)
        """
        print(f"\n{'='*60}")
        print(f"🎥 APERTURA WEBCAM {camera_id}")
        print(f"{'='*60}")
        
        # Step 1: Prova ad aprire la webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Errore: impossibile aprire la webcam {camera_id}")
            print(f"\n🔧 Troubleshooting:")
            print(f"  - Verifica che la webcam sia collegata")
            print(f"  - Prova con camera_id diverso: python run-webcam.py 1")
            print(f"  - Su Linux potrebbe essere /dev/video0 invece di 0")
            print(f"  - Controlla permessi della webcam")
            raise RuntimeError(f"Impossibile aprire webcam {camera_id}")
        
        # Step 2: Configura la webcam
        print(f"✓ Webcam {camera_id} aperta")
        print(f"⚙️ Configurazione webcam...")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Step 3: Attendi che la webcam si inizializzi
        print(f"⏳ Inizializzazione camera (5 tentativi)...")
        frame = None
        for i in range(5):
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"  Tentativo {i+1}: ✓ Frame ricevuto")
                frame = test_frame
                break
            else:
                print(f"  Tentativo {i+1}: ✗ Nessun frame")
            time.sleep(0.3)
        
        if frame is None:
            print(f"❌ Webcam non ha risposto dopo 5 tentativi")
            print(f"\n🔧 Troubleshooting:")
            print(f"  - La webcam potrebbe essere in uso da un'altra app")
            print(f"  - Prova a riavviare la webcam")
            print(f"  - Prova a riavviare il sistema")
            cap.release()
            raise RuntimeError(f"Webcam {camera_id} non risponde")
        
        # Step 4: Debug info
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
                
                # Diagnostica ogni 30 frame
                if frame_count % 30 == 0:
                    if frame.max() < 10:
                        print(f"⚠️ Frame {frame_count}: Ancora nero (max pixel={frame.max()})")
                
                # Flip orizzontale per webcam (effetto specchio)
                frame = cv2.flip(frame, 1)
                
                # Esegui detection
                annotated_frame, detections = self.detect_frame(frame)
                
                # Aggiungi info
                annotated_frame = self.draw_info(annotated_frame, detections, fps)
                
                # Mostra il frame
                cv2.imshow("Fire Detection - Webcam", annotated_frame)
                
                # Calcola FPS
                fps_counter += 1
                if time.time() - fps_time > 1:
                    fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()
                
                # Gestisci input
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:  # 'q' o ESC
                    break
                elif key == ord('s'):
                    self.save_frame(annotated_frame, detections)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n✓ Detection da webcam terminata ({frame_count} frame elaborati)")
    
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
                cv2.imshow("Fire Detection - RTMP Stream", annotated_frame)
                
                # Calcola FPS
                fps_counter += 1
                if time.time() - fps_time > 1:
                    fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()
                
                # Gestisci input
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:  # 'q' o ESC
                    break
                elif key == ord('s'):
                    self.save_frame(annotated_frame, detections)
        
        except KeyboardInterrupt:
            print("\n\nInterrotto dall'utente")
        finally:
            cap.release()
            cv2.destroyAllWindows()
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
                cv2.imshow("Fire Detection - Video File", annotated_frame)
                
                # Calcola FPS
                fps_counter += 1
                if time.time() - fps_time > 1:
                    fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()
                
                # Gestisci input
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:  # 'q' o ESC
                    break
                elif key == ord('s'):
                    self.save_frame(annotated_frame, detections)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
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
            cv2.imshow("Fire Detection - Test Images", annotated_img)
            
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
        
        cv2.destroyAllWindows()
        
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
  python detect.py                              # Webcam (default)
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
        default="0",
        help="Sorgente: numero=webcam, RTMP/RTSP URL=stream, path file=video, path cartella=immagini"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/fire_detector_runs/train/weights/best.pt",
        help="Percorso del modello YOLOv8 (creato da train.py)"
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
        
        # Se è un numero, è la webcam
        if source.isdigit():
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
            exit(1)
    
    except FileNotFoundError as e:
        print(f"\n❌ Errore: {e}")
        exit(1)
    except RuntimeError as e:
        print(f"\n❌ Errore: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\n\n✓ Interrotto dall'utente")


if __name__ == "__main__":
    main()
