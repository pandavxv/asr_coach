# Vietnamese Presentation ASR Coach

He thong ho tro luyen thuyet trinh tieng Viet dua tren nhan dang tieng noi tu dong, su dung `vinai/PhoWhisper-base`.

## Scope

- Record truc tiep hoac upload audio bai thuyet trinh.
- Chuyen speech to text bang PhoWhisper-base.
- Hien thi transcript.
- Tinh thoi luong, so tu, WPM va filler words tieng Viet.
- Tinh WER/CER neu co transcript chuan.
- Tao feedback tong ket don gian.

## Tech stack

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- Librosa/SoundFile

## Cai dat

```powershell
conda create -y -n asr_coach python=3.11 pip
conda activate asr_coach
conda install -y -c conda-forge liblzma
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Lenh tren tao conda env `asr_coach` va cai PyTorch CUDA wheel rieng cho NVIDIA GPU. Khong can cai CUDA Toolkit he thong de chay demo nay, mien la `nvidia-smi` nhan GPU va driver du moi.

Kiem tra GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Kiem tra Python co du runtime nen `lzma` cho Transformers:

```powershell
python -c "import lzma; print('lzma ok')"
```

Neu gap loi `ImportError: DLL load failed while importing _lzma`, moi truong Python dang thieu `xz/liblzma`. Cach sach nhat la tao lai env bang conda:

```powershell
conda deactivate
conda env remove -n asr_coach
conda create -y -n asr_coach python=3.11 pip
conda activate asr_coach
conda install -y -c conda-forge liblzma
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Neu muon sua env hien tai truoc khi cai lai dependencies:

```powershell
conda install -n asr_coach -y -c conda-forge liblzma
python -c "import lzma; print('lzma ok')"
```

Tren mot so moi truong Windows co the gap loi OpenMP trung `libiomp5md.dll` khi import PyTorch/Transformers. App dat `KMP_DUPLICATE_LIB_OK=TRUE` trong process de phuc vu demo local; neu lam ban nop chinh thuc, nen cai PyTorch/NumPy/Librosa trong mot virtual environment moi.

## Chay app

```powershell
streamlit run app.py
```

Lan dau chay, model `vinai/PhoWhisper-base` se duoc tai ve tu Hugging Face va cache tren may.

App ho tro 2 nguon audio:

- `Record truc tiep`: ghi am tu microphone trong trinh duyet va phan tich ngay.
- `Upload file`: tai len file `.wav`, `.mp3`, `.m4a`, `.flac` hoac `.ogg`.

## Chay test nhanh

```powershell
python -m unittest discover
```

## Fine-tune Nhe Tren Colab

Repo co notebook fine-tune LoRA cuc nhe tren VietSuperSpeech:

```text
notebooks/finetune_phowhisper_lora_vietsuperspeech.ipynb
```

Huong dan chi tiet nam trong:

```text
training/README.md
```

Mac dinh notebook fine-tune `vinai/PhoWhisper-base` voi 300 train samples, 60 eval samples va 100 steps de phu hop demo mon hoc.

## Cau truc

```text
app.py
presentation_coach/
  asr.py
  audio.py
  analysis.py
  evaluation.py
  report.py
training/
  finetune_lora_vietsuperspeech.py
  colab_requirements.txt
tests/
  test_analysis.py
  test_evaluation.py
  test_report.py
```
