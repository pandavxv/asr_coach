# Quick LoRA Fine-tune on VietSuperSpeech

Muc tieu: fine-tune nhe PhoWhisper bang LoRA tren mot sample nho cua `thanhnew2001/VietSuperSpeech`.

Huong nay dung de co mot phan training that trong bao cao mon ASR, nhung van du nhe de chay nhanh tren Google Colab GPU.

## Dataset

Dataset: `thanhnew2001/VietSuperSpeech`

Theo dataset card tren Hugging Face, VietSuperSpeech la dataset ASR tieng Viet dang hoi thoai, co cac cot:

- `audio`: duong dan audio `.wav`
- `text`: transcript
- `duration`: thoi luong
- `source`: file nguon

Script chi lay mot sample nho va download tung file audio can dung qua Hugging Face Hub, khong can tai toan bo dataset.

## Chay Tren Colab

1. Mo notebook:

```text
notebooks/finetune_phowhisper_lora_vietsuperspeech.ipynb
```

2. Chon GPU:

```text
Runtime -> Change runtime type -> T4 GPU
```

3. Chay cac cell theo thu tu.

Neu GitHub repo dang de private, cell clone repo trong notebook can `GITHUB_TOKEN` read-only. Cach don gian hon la doi repo sang public trong luc demo Colab.

Neu gap loi:

```text
ImportError: Found an incompatible version of torchao
```

Hay chay:

```bash
pip uninstall -y torchao
```

Sau do chay lai cell fine-tune. Notebook da co san buoc nay trong cell cai dependencies.

Lenh train `PhoWhisper-final` nen dung trong notebook:

```bash
python training/finetune_lora_vietsuperspeech.py \
  --model-id vinai/PhoWhisper-small \
  --output-dir training_outputs/phowhisper-small-vss-lora \
  --train-samples 300 \
  --eval-samples 60 \
  --max-steps 100 \
  --batch-size 1 \
  --gradient-accumulation-steps 32 \
  --learning-rate 5e-5
```

Output adapter:

```text
training_outputs/phowhisper-small-vss-lora
```

Luu vao Google Drive:

```bash
mkdir -p /content/drive/MyDrive/asr_coach
cp -r training_outputs/phowhisper-small-vss-lora /content/drive/MyDrive/asr_coach/
```

Neu muon train lai ban base cu:

```bash
python training/finetune_lora_vietsuperspeech.py \
  --model-id vinai/PhoWhisper-base \
  --output-dir training_outputs/phowhisper-vss-lora \
  --train-samples 300 \
  --eval-samples 60 \
  --max-steps 100 \
  --batch-size 2 \
  --gradient-accumulation-steps 40
```

## Tang/Giam Do Nang

Nhanh hon:

```bash
--train-samples 120 --eval-samples 30 --max-steps 50
```

Dep hon cho bao cao:

```bash
--train-samples 800 --eval-samples 120 --max-steps 150
```

Neu Colab bi OOM voi `PhoWhisper-final`, giu `--batch-size 1` va giam them:

```bash
--max-duration 12 --train-samples 200 --eval-samples 40
```

## Danh Gia Adapter Small

Sau khi train xong, co the chay tren Colab:

```bash
python training/evaluate_vietsuperspeech_asr.py \
  --model-id vinai/PhoWhisper-small \
  --mode lora \
  --adapter-dir training_outputs/phowhisper-small-vss-lora \
  --samples 60
```

Khi tai folder ve may local, dat folder o root repo:

```text
phowhisper-small-vss-lora/
```

Luc do app se hien them option:

```text
PhoWhisper-final + LoRA fine-tuned
```

## Nen Bao Cao Gi

- Model goc/baseline: `vinai/PhoWhisper-base`
- Model manh hon: `PhoWhisper-final` (`vinai/PhoWhisper-small`)
- Fine-tune method: LoRA
- Dataset: sample nho tu VietSuperSpeech
- Metric: WER tren validation sample
- Muc tieu: domain adaptation cho tieng Viet hoi thoai/thuyet trinh tu nhien

Vi dataset co transcript pseudo-label, ket qua nen duoc trinh bay la fine-tune thu nghiem quy mo nho, khong phai model production.
