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
    - For image testing: Use arrow keys to navigate
"""

from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from datetime import datetime


class FireDetector:
    """Fire detection system using YOLOv8"""
    
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
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Errore: impossibile aprire la webcam {camera_id}")
        
        # Configura la webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*60)
        print("DETECTION DA WEBCAM")
        print("="*60)
        print("Premi 'q' o 'ESC' per uscire")
        print("Premi 's' per salvare il frame")
        print("="*60 + "\n")
        
        import time
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Errore: impossibile leggere il frame")
                    break
                
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
            print("\n✓ Detection da webcam terminata")
    
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
        print("  ← → : Naviga tra le immagini")
        print("  's' : Salva immagine corrente")
        print("  'q' o ESC: Esci")
        print("="*60 + "\n")
        
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
            
            # Gestisci input
            key = cv2.waitKey(0) & 0xFF
            
            if key in [ord('q'), 27]:  # 'q' o ESC
                break
            elif key == ord('s'):  # Salva
                self.save_frame(annotated_img, detections)
            elif key == 81:  # Freccia sinistra (←)
                current_idx = (current_idx - 1) % len(image_paths)
            elif key == 83:  # Freccia destra (→)
                current_idx = (current_idx + 1) % len(image_paths)
        
        cv2.destroyAllWindows()
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
