"""
complete_mp3_to_srt.py - All-in-one MP3 to SRT Converter with CUDA Support

This script provides:
1. Select audio file (MP3, WAV, etc.)
2. Create virtual environment and install required packages (with CUDA support)
3. Convert audio to subtitle (SRT) with GPU acceleration

Usage: python complete_mp3_to_srt.py
"""

import os
import sys
import threading
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple
import re

# Optional modules
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False

try:
    import venv
    VENV_AVAILABLE = True
except ImportError:
    VENV_AVAILABLE = False

# ==================== Helper Functions ====================

def is_windows() -> bool:
    return os.name == 'nt'

def python_path_for_venv(venv_path: str) -> str:
    if is_windows():
        return os.path.join(venv_path, 'Scripts', 'python.exe')
    else:
        return os.path.join(venv_path, 'bin', 'python')

def is_valid_venv(venv_path: str) -> bool:
    """Check if the path contains a valid virtual environment"""
    py_path = python_path_for_venv(venv_path)
    return os.path.exists(py_path)

def run_subprocess(cmd: List[str], cwd: str = None) -> Tuple[int, str]:
    try:
        completed = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT, text=True)
        return completed.returncode, completed.stdout
    except Exception as e:
        return 1, str(e)

def detect_cuda() -> bool:
    """Detect if CUDA is available on the system"""
    try:
        # Try nvidia-smi command
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def create_venv_and_install(venv_path: str, packages: List[str], use_cuda: bool = False, console_log_fn=print) -> str:
    """Create venv and install packages. Returns the Python executable path."""
    venv_path = os.path.abspath(venv_path)
    console_log_fn(f'Environment path: {venv_path}')
    
    base_dir = os.path.dirname(venv_path) or '.'
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    
    use_venv = VENV_AVAILABLE
    
    if use_venv:
        if os.path.exists(venv_path) and is_valid_venv(venv_path):
            console_log_fn(f'Using existing virtual environment at: {venv_path}')
        else:
            if os.path.exists(venv_path) and not is_valid_venv(venv_path):
                console_log_fn(f'Warning: Folder exists but is not a valid virtual environment.')
                console_log_fn(f'Creating new virtual environment at: {venv_path}')
            else:
                console_log_fn(f'Creating virtual environment at: {venv_path} ...')
            
            os.makedirs(venv_path, exist_ok=True)
            try:
                builder = venv.EnvBuilder(with_pip=True, clear=True)
                builder.create(venv_path)
                console_log_fn('Virtual environment created successfully.')
            except Exception as e:
                console_log_fn(f'Error creating virtual environment: {e}')
                console_log_fn('Falling back to system Python.')
                py_path = sys.executable
                use_venv = False
    
    if use_venv:
        py_path = python_path_for_venv(venv_path)
        if not os.path.exists(py_path):
            console_log_fn(f'Error: Python executable not found at {py_path}')
            console_log_fn('Falling back to system Python.')
            py_path = sys.executable
    else:
        console_log_fn('Warning: venv module not available. Using system Python.')
        py_path = sys.executable
    
    console_log_fn(f'Using Python: {py_path}')
    
    # Upgrade pip first
    console_log_fn('Upgrading pip...')
    code, out = run_subprocess([py_path, '-m', 'pip', 'install', '--upgrade', 'pip'])
    if code == 0:
        console_log_fn('pip upgraded successfully.')
    else:
        console_log_fn(f'Warning: Could not upgrade pip.\n{out}')
    
    # Install PyTorch with CUDA support if requested
    if use_cuda:
        console_log_fn('\n=== Installing PyTorch with CUDA 11.8 support ===')
        console_log_fn('This may take a few minutes...')
        console_log_fn('IMPORTANT: Uninstalling any existing PyTorch first...')
        
        # Uninstall existing PyTorch to avoid conflicts
        run_subprocess([py_path, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio'])
        
        console_log_fn('Installing: torch==2.8.0+cu128, torchvision==0.23.0+cu128, torchaudio==2.8.0')
        torch_cmd = [
            py_path, '-m', 'pip', 'install', 
            'torch==2.8.0+cu128', 'torchvision==0.23.0+cu128', 'torchaudio==2.8.0',
            '--index-url', 'https://download.pytorch.org/whl/cu128', '--no-cache-dir'
        ]
        code, out = run_subprocess(torch_cmd)
        console_log_fn(out)
        if code != 0:
            console_log_fn('Failed to install PyTorch with CUDA. Falling back to CPU version.')
            code, out = run_subprocess([py_path, '-m', 'pip', 'install', 'torch==2.0.1', 'torchvision==0.15.2', 'torchaudio==2.0.2', '--no-cache-dir'])
            console_log_fn(out)
        else:
            console_log_fn('[OK] PyTorch with CUDA installed successfully.')
            # Install compatible transformers version
            console_log_fn('\nInstalling transformers==4.30.2 (compatible with PyTorch 2.0.1)...')
            code, out = run_subprocess([py_path, '-m', 'pip', 'install', 'transformers==4.30.2'])
            console_log_fn(out)
    
    # Install other packages
    for pkg in packages:
        if 'torch' in pkg.lower() and use_cuda:
            console_log_fn(f'Skipping {pkg} (already installed with CUDA support)')
            continue
        if 'transformers' in pkg.lower() and use_cuda:
            console_log_fn(f'Skipping {pkg} (already installed compatible version)')
            continue
            
        console_log_fn(f'Installing {pkg} ...')
        code, out = run_subprocess([py_path, '-m', 'pip', 'install', pkg])
        console_log_fn(out)
        if code != 0:
            console_log_fn(f'Failed to install {pkg}. See output above.')
        else:
            console_log_fn(f'Successfully installed {pkg}.')
    
    console_log_fn('\n=== Environment setup complete ===')
    return py_path

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def split_segments_to_sentences(segments):
    """
    Tách segments thành các câu riêng lẻ dựa trên dấu câu tiếng Nhật
    Hỗ trợ: 。!?...
    """
    new_segments = []
    
    for seg in segments:
        text = seg['text'].strip()
        
        sentences = re.split(r'(?<=[。!?...])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) == 0:
            new_segments.append(seg)
            continue

        if len(sentences) == 1:
            new_segments.append(seg)
            continue

        total_chars = sum(len(s) for s in sentences)
        duration_total = seg['end'] - seg['start']
        
        current_time = seg['start']
        
        for i, sentence in enumerate(sentences):
            if i == len(sentences) - 1:
                sentence_duration = seg['end'] - current_time
            else:
                char_ratio = len(sentence) / total_chars
                sentence_duration = duration_total * char_ratio
            
            end_time = current_time + sentence_duration
            
            new_segments.append({
                'start': current_time,
                'end': end_time,
                'text': sentence
            })
            
            current_time = end_time
    
    return new_segments

def write_srt(segments, output_path):
    """Write segments to SRT file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            text = seg['text'].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

# ==================== GUI Application ====================

if TK_AVAILABLE:
    class CompleteApp:
        def __init__(self, root):
            self.root = root
            root.title("Complete MP3 → SRT Converter (CUDA Optimized)")
            root.geometry("950x850")
            
            # Detect CUDA
            self.cuda_available = detect_cuda()
            
            # Create notebook (tabs)
            notebook = ttk.Notebook(root)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Tab 1: Setup Environment
            setup_frame = tk.Frame(notebook)
            notebook.add(setup_frame, text="1️⃣ Setup Environment")
            self.create_setup_tab(setup_frame)
            
            # Tab 2: Convert to SRT
            convert_frame = tk.Frame(notebook)
            notebook.add(convert_frame, text="2️⃣ Convert to SRT")
            self.create_convert_tab(convert_frame)
            
            self.python_executable = None
        
        def create_setup_tab(self, parent):
            frm = tk.Frame(parent, padx=20, pady=20)
            frm.pack(fill=tk.BOTH, expand=True)
            
            # Info label
            info = tk.Label(frm, text="Step 1: Create virtual environment and install required packages",
                          font=("", 11, "bold"), fg="blue")
            info.pack(pady=(0,10))
            
            # Important note for reinstallation
            note_frame = tk.Frame(frm, relief=tk.RIDGE, borderwidth=2, bg="#FFF9C4", padx=10, pady=10)
            note_frame.pack(fill=tk.X, pady=(0,15))
            tk.Label(note_frame, text="⚠️ If you have an existing environment with errors:", 
                    font=("", 9, "bold"), bg="#FFF9C4").pack(anchor='w')
            tk.Label(note_frame, text="1. Delete the venv folder manually", 
                    font=("", 8), bg="#FFF9C4").pack(anchor='w', padx=(15,0))
            tk.Label(note_frame, text="2. Click 'Create venv & Install packages' to reinstall fresh", 
                    font=("", 8), bg="#FFF9C4").pack(anchor='w', padx=(15,0))
            
            # CUDA status
            cuda_frame = tk.Frame(frm, relief=tk.RIDGE, borderwidth=2, padx=10, pady=10)
            cuda_frame.pack(fill=tk.X, pady=(0,20))
            
            if self.cuda_available:
                tk.Label(cuda_frame, text="🎮 CUDA Status: DETECTED ✓", 
                        font=("", 10, "bold"), fg="green").pack(anchor='w')
                tk.Label(cuda_frame, text="GPU acceleration will be available for faster processing",
                        font=("", 9)).pack(anchor='w')
            else:
                tk.Label(cuda_frame, text="⚠️ CUDA Status: NOT DETECTED", 
                        font=("", 10, "bold"), fg="orange").pack(anchor='w')
                tk.Label(cuda_frame, text="CPU mode will be used (slower). Install NVIDIA drivers to enable GPU.",
                        font=("", 9)).pack(anchor='w')
            
            # CUDA checkbox
            self.use_cuda_var = tk.BooleanVar(value=self.cuda_available)
            cuda_check = tk.Checkbutton(frm, text="Install PyTorch with CUDA support (for RTX 3060 Ti)", 
                                       variable=self.use_cuda_var, font=("", 10, "bold"))
            cuda_check.pack(anchor='w', pady=(0,15))
            if not self.cuda_available:
                cuda_check.config(state='disabled')
            
            # Environment folder
            tk.Label(frm, text='Environment folder:', font=("", 10, "bold")).pack(anchor='w')
            env_frame = tk.Frame(frm)
            env_frame.pack(fill=tk.X, pady=(5,15))
            self.env_var = tk.StringVar(value='./venv')
            tk.Entry(env_frame, textvariable=self.env_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Button(env_frame, text='Choose folder', command=self.choose_env_dir).pack(side=tk.LEFT, padx=(5,0))
            
            # Packages to install
            tk.Label(frm, text='Packages to install (comma-separated):', font=("", 10, "bold")).pack(anchor='w')
            default_pkgs = 'whisperx, ffmpeg-python'
            if not self.cuda_available:
                default_pkgs += ', torch'
            self.pkgs_var = tk.StringVar(value=default_pkgs)
            tk.Entry(frm, textvariable=self.pkgs_var, width=70).pack(fill=tk.X, pady=(5,15))
            
            # Install button
            self.setup_btn = tk.Button(frm, text='🔧 Create venv & Install packages',
                                      command=self.on_setup, font=("", 11, "bold"),
                                      bg="#2196F3", fg="white", padx=20, pady=10)
            self.setup_btn.pack(pady=(0,15))
            
            # Console output
            tk.Label(frm, text='Console output:', font=("", 10, "bold")).pack(anchor='w')
            self.setup_console = scrolledtext.ScrolledText(frm, width=90, height=18, font=("Consolas", 9))
            self.setup_console.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        
        def create_convert_tab(self, parent):
            frm = tk.Frame(parent, padx=20, pady=20)
            frm.pack(fill=tk.BOTH, expand=True)
            
            # Info label
            info = tk.Label(frm, text="Step 2: Select audio file and convert to subtitle",
                          font=("", 11, "bold"), fg="green")
            info.pack(pady=(0,20))
            
            # Audio file selection
            tk.Label(frm, text="Audio file:", font=("", 10, "bold")).pack(anchor='w')
            audio_frame = tk.Frame(frm)
            audio_frame.pack(fill=tk.X, pady=(5,15))
            self.audio_var = tk.StringVar()
            tk.Entry(audio_frame, textvariable=self.audio_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Button(audio_frame, text="Browse", command=self.select_audio).pack(side=tk.LEFT, padx=(5,0))
            
            # Output file
            tk.Label(frm, text="Output SRT file (optional - auto-generated if empty):", font=("", 10, "bold")).pack(anchor='w')
            output_frame = tk.Frame(frm)
            output_frame.pack(fill=tk.X, pady=(5,15))
            self.output_var = tk.StringVar()
            tk.Entry(output_frame, textvariable=self.output_var, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Button(output_frame, text="Browse", command=self.select_output).pack(side=tk.LEFT, padx=(5,0))
            
            # Settings frame
            settings_frame = tk.LabelFrame(frm, text="Settings (Optimized for RTX 3060 Ti)", padx=10, pady=10)
            settings_frame.pack(fill=tk.X, pady=(0,15))
            
            row1 = tk.Frame(settings_frame)
            row1.pack(fill=tk.X, pady=5)
            tk.Label(row1, text="Language:").pack(side=tk.LEFT, padx=(0,5))
            self.lang_var = tk.StringVar(value="auto")
            lang_combo = ttk.Combobox(row1, textvariable=self.lang_var, width=12, state='readonly')
            lang_combo['values'] = ("auto", "en", "vi", "ja", "zh", "ko", "fr", "de", "es", "ru", "th", "id")
            lang_combo.pack(side=tk.LEFT, padx=(0,20))
            
            tk.Label(row1, text="Model size:").pack(side=tk.LEFT, padx=(0,5))
            self.model_var = tk.StringVar(value="base")
            model_combo = ttk.Combobox(row1, textvariable=self.model_var, width=12, state='readonly')
            model_combo['values'] = ("tiny", "base", "small", "medium", "large-v2", "large-v3")
            model_combo.pack(side=tk.LEFT)
            
            row2 = tk.Frame(settings_frame)
            row2.pack(fill=tk.X, pady=5)
            tk.Label(row2, text="Device:").pack(side=tk.LEFT, padx=(0,5))
            self.device_var = tk.StringVar(value="cuda" if self.cuda_available else "cpu")
            device_combo = ttk.Combobox(row2, textvariable=self.device_var, width=12, state='readonly')
            device_combo['values'] = ("cpu", "cuda")
            device_combo.pack(side=tk.LEFT, padx=(0,20))
            
            tk.Label(row2, text="Compute type:").pack(side=tk.LEFT, padx=(0,5))
            # RTX 3060 Ti supports float16 for better performance
            default_compute = "float16" if self.cuda_available else "float32"
            self.compute_var = tk.StringVar(value=default_compute)
            compute_combo = ttk.Combobox(row2, textvariable=self.compute_var, width=12, state='readonly')
            compute_combo['values'] = ("float32", "float16", "int8")
            compute_combo.pack(side=tk.LEFT)
            
            row3 = tk.Frame(settings_frame)
            row3.pack(fill=tk.X, pady=5)
            tk.Label(row3, text="Batch size:").pack(side=tk.LEFT, padx=(0,5))
            # RTX 3060 Ti with 8GB VRAM can handle larger batches
            default_batch = "32" if self.cuda_available else "16"
            self.batch_var = tk.StringVar(value=default_batch)
            batch_combo = ttk.Combobox(row3, textvariable=self.batch_var, width=12, state='readonly')
            batch_combo['values'] = ("8", "16", "32", "64")
            batch_combo.pack(side=tk.LEFT)
            tk.Label(row3, text="(Higher = faster but more VRAM)", font=("", 8), fg="gray").pack(side=tk.LEFT, padx=(10,0))
            
            # Convert button
            self.convert_btn = tk.Button(frm, text="🎬 Convert to SRT",
                                        command=self.start_conversion, font=("", 11, "bold"),
                                        bg="#4CAF50", fg="white", padx=20, pady=10)
            self.convert_btn.pack(pady=(0,15))
            
            # Progress bar
            self.progress = ttk.Progressbar(frm, mode='indeterminate')
            self.progress.pack(fill=tk.X, pady=(0,10))
            
            # Console output
            tk.Label(frm, text="Console output:", font=("", 10, "bold")).pack(anchor='w')
            self.convert_console = scrolledtext.ScrolledText(frm, width=90, height=12, font=("Consolas", 9))
            self.convert_console.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        
        # Setup tab methods
        def choose_env_dir(self):
            folder = filedialog.askdirectory()
            if folder:
                self.env_var.set(folder)
                self.log_setup(f'Environment path set to: {folder}')
        
        def log_setup(self, text: str):
            self.setup_console.insert(tk.END, text + '\n')
            self.setup_console.see(tk.END)
            self.root.update()
        
        def on_setup(self):
            self.setup_btn.config(state='disabled')
            t = threading.Thread(target=self._setup_thread)
            t.start()
        
        def _setup_thread(self):
            try:
                env_path = self.env_var.get().strip() or './venv'
                pkgs = [p.strip() for p in (self.pkgs_var.get() or '').split(',') if p.strip()]
                use_cuda = self.use_cuda_var.get()
                
                self.python_executable = create_venv_and_install(env_path, pkgs, use_cuda=use_cuda, console_log_fn=self.log_setup)
                self.log_setup(f'\n✓ Python executable: {self.python_executable}')
                self.log_setup('\n=== All done! Now go to "Convert to SRT" tab ===')
                messagebox.showinfo("Success", "Environment setup complete!\n\nNow switch to the 'Convert to SRT' tab.")
            except Exception as e:
                self.log_setup(f'Error: {e}')
                import traceback
                self.log_setup(traceback.format_exc())
            finally:
                self.setup_btn.config(state='normal')
        
        # Convert tab methods
        def select_audio(self):
            fn = filedialog.askopenfilename(
                title="Select audio file",
                filetypes=[
                    ('Audio files', '*.mp3 *.wav *.m4a *.flac *.ogg *.wma'),
                    ('All files', '*.*')
                ]
            )
            if fn:
                self.audio_var.set(fn)
                output = str(Path(fn).with_suffix('.srt'))
                self.output_var.set(output)
                self.log_convert(f"Selected: {fn}")
        
        def select_output(self):
            fn = filedialog.asksaveasfilename(
                title="Save SRT file as",
                defaultextension=".srt",
                filetypes=[('SRT files', '*.srt'), ('All files', '*.*')]
            )
            if fn:
                self.output_var.set(fn)
                self.log_convert(f"Output: {fn}")
        
        def log_convert(self, text):
            self.convert_console.insert(tk.END, text + '\n')
            self.convert_console.see(tk.END)
            self.root.update()
        
        def start_conversion(self):
            audio_path = self.audio_var.get().strip()
            if not audio_path:
                messagebox.showwarning("Warning", "Please select an audio file!")
                return
            
            if not os.path.exists(audio_path):
                messagebox.showerror("Error", f"File not found: {audio_path}")
                return
            
            self.convert_btn.config(state='disabled')
            self.progress.start()
            thread = threading.Thread(target=self._convert_thread)
            thread.start()
        
        def _convert_thread(self):
            try:
                env_path = self.env_var.get().strip()
                if env_path and is_valid_venv(env_path):
                    py_executable = python_path_for_venv(env_path)
                    self.log_convert(f"Using Python from virtual environment: {py_executable}\n")
                    
                    # Get parameters
                    audio_path = self.audio_var.get().strip()
                    output_path = self.output_var.get().strip()
                    language = self.lang_var.get()
                    model_size = self.model_var.get()
                    device = self.device_var.get()
                    compute_type = self.compute_var.get()
                    batch_size = self.batch_var.get()
                    
                    self.log_convert("="*70)
                    self.log_convert("Starting conversion using virtual environment...")
                    if device == "cuda":
                        self.log_convert("[GPU] GPU acceleration enabled (CUDA)")
                        self.log_convert(f"[GPU] Optimized for RTX 3060 Ti: {compute_type}, batch_size={batch_size}")
                    self.log_convert("="*70 + "\n")
                    
                    # Create conversion script with CUDA optimization and Windows encoding fix
                    script_content = f'''# -*- coding: utf-8 -*-
import sys
import io
import re

# Fix Windows console encoding issues
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, r"{env_path}")
from pathlib import Path
import torch

# Verify CUDA availability
print("[INFO] PyTorch version:", torch.__version__)
print("[INFO] CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[INFO] CUDA version:", torch.version.cuda)
    print("[INFO] Number of GPUs:", torch.cuda.device_count())

# Fix PyTorch weights_only issue - more robust approach
import torch.serialization
# Add safe globals for WhisperX dependencies
try:
    from omegaconf import DictConfig, ListConfig
    torch.serialization.add_safe_globals([DictConfig, ListConfig])
except:
    pass

_original_load = torch.load
def _patched_load(*args, **kwargs):
    # Force weights_only=False for PyTorch 2.6+
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# Display GPU info if CUDA is available
if torch.cuda.is_available():
    print(f"[GPU] GPU: {{torch.cuda.get_device_name(0)}}")
    print(f"[GPU] VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}} GB")
    print(f"[GPU] CUDA ready for processing")
    print()
else:
    print("[WARNING] CUDA not available, using CPU")
    print("[WARNING] This will be much slower than GPU processing")
    print()

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{{hours:02d}}:{{minutes:02d}}:{{secs:02d}},{{millis:03d}}"

def split_segments_to_sentences(segments):
    new_segments = []
    
    for seg in segments:
        text = seg['text'].strip()
        
        sentences = re.split(r'(?<=[。!?...])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) == 0:
            new_segments.append(seg)
            continue

        if len(sentences) == 1:
            new_segments.append(seg)
            continue

        total_chars = sum(len(s) for s in sentences)
        duration_total = seg['end'] - seg['start']
        
        current_time = seg['start']
        
        for i, sentence in enumerate(sentences):
            if i == len(sentences) - 1:
                sentence_duration = seg['end'] - current_time
            else:
                char_ratio = len(sentence) / total_chars
                sentence_duration = duration_total * char_ratio
            
            end_time = current_time + sentence_duration
            
            new_segments.append({{
                'start': current_time,
                'end': end_time,
                'text': sentence
            }})
            
            current_time = end_time
    
    return new_segments

def write_srt(segments, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            text = seg['text'].strip()
            f.write(f"{{i}}\\n{{start}} --> {{end}}\\n{{text}}\\n\\n")

print("Loading WhisperX...")
import whisperx

audio_path = r"{audio_path}"
output_path_str = r"{output_path}"
output_path = output_path_str if output_path_str else None
language = "{language}" if "{language}" != "auto" else None
model_size = "{model_size}"
device = "{device}"
compute_type = "{compute_type}"
batch_size = {batch_size}

# Auto-adjust settings if CUDA not available
if device == "cuda" and not torch.cuda.is_available():
    print("[WARNING] CUDA requested but not available, falling back to CPU")
    device = "cpu"
    compute_type = "float32"
    batch_size = 8
    print(f"[INFO] Adjusted settings: device={{device}}, compute_type={{compute_type}}, batch_size={{batch_size}}")

if output_path is None or output_path == "":
    audio_file = Path(audio_path)
    output_path = str(audio_file.with_suffix('.srt'))

print(f"Audio file: {{audio_path}}")
print(f"Output file: {{output_path}}")
print(f"Model: {{model_size}}, Device: {{device}}, Compute: {{compute_type}}")
print(f"Batch size: {{batch_size}}")
print()

print("Loading audio...")
audio = whisperx.load_audio(audio_path)

print(f"Loading Whisper model ({{model_size}})...")
model = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)

print("Transcribing audio...")
result = model.transcribe(audio, batch_size=batch_size)

if language is None:
    detected_lang = result.get("language", "unknown")
    print(f"Detected language: {{detected_lang}}")

print("Aligning timestamps...")
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], 
    device=device
)
result = whisperx.align(
    result["segments"], 
    model_a, 
    metadata, 
    audio, 
    device,
    return_char_alignments=False
)

print("Writing SRT file...")
segments = result["segments"]
print("Spliting SRT file...")
segments = split_segments_to_sentences(segments)
write_srt(segments, output_path)

print(f"\\n[SUCCESS] Successfully created: {{output_path}}")
print(f"[SUCCESS] Total segments: {{len(segments)}}")

# Clean up GPU memory
if device == "cuda":
    del model
    del model_a
    torch.cuda.empty_cache()
    print("[GPU] GPU memory cleared")
'''
                    
                    # Write and run temporary script
                    temp_script = os.path.join(os.path.dirname(env_path), 'temp_convert.py')
                    with open(temp_script, 'w', encoding='utf-8') as f:
                        f.write(script_content)
                    
                    try:
                        process = subprocess.Popen(
                            [py_executable, temp_script],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            encoding='utf-8',
                            errors='replace'
                        )
                        
                        for line in process.stdout:
                            self.log_convert(line.rstrip())
                        
                        process.wait()
                        
                        if process.returncode == 0:
                            self.log_convert("\n" + "="*70)
                            self.log_convert("[SUCCESS] CONVERSION COMPLETED SUCCESSFULLY!")
                            self.log_convert("="*70)
                            output_file = output_path if output_path else str(Path(audio_path).with_suffix('.srt'))
                            messagebox.showinfo("Success", f"Subtitle created!\n\n{output_file}")
                        else:
                            self.log_convert("\n" + "="*70)
                            self.log_convert("[ERROR] CONVERSION FAILED")
                            self.log_convert("="*70)
                            messagebox.showerror("Error", "Conversion failed. Check console output.")
                    finally:
                        if os.path.exists(temp_script):
                            os.remove(temp_script)
                else:
                    self.log_convert("Error: Virtual environment not found or not valid!")
                    self.log_convert("Please go to 'Setup Environment' tab and create the environment first.")
                    messagebox.showerror("Error", "Virtual environment not found!\n\nPlease setup environment first in Tab 1.")
            
            except Exception as e:
                self.log_convert(f"\nUnexpected error: {e}")
                import traceback
                self.log_convert(traceback.format_exc())
                messagebox.showerror("Error", str(e))
            finally:
                self.progress.stop()
                self.convert_btn.config(state='normal')

    def main():
        root = tk.Tk()
        app = CompleteApp(root)
        root.mainloop()

else:
    def main():
        print("GUI not available. Please install tkinter.")

if __name__ == '__main__':
    main()