# Дообучение LoRA (минимальный шаблон)

Минимальный пайплайн для дообучения открытой текст‑в‑изображение модели (Stable Diffusion 1.5) с помощью LoRA‑адаптеров и генерации изображений. Подходит для встройки в веб‑сервис.

## Что входит
- Предобработка пользовательских изображений
- Скрипт обучения LoRA (DreamBooth‑подобный) на  Diffusers + PEFT
- Скрипт инференса с подключением сохранённого LoRA
- Рассчитано на потребительские GPU (≈8–12 ГБ) при 512×512

## Установка
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Подготовка данных
Положите 8–15 изображений одного стиля в каталог, например `data/my_style/`.
Подписи к изображениям берутся в порядке приоритета:
1) `image.jpg` + файл `image.txt` с подписью
2) `captions.txt` в формате `filename<TAB>caption`
3) шаблон: `a photo of <{placeholder_token}>`

## Обучение
```bash
python src/train_lora.py \
  --train_data_dir data/my_style \
  --output_dir outputs/my_style \
  --placeholder_token mystyle \
  --resolution 512 \
  --batch_size 2 \
  --epochs 10
```
Модель и LoRA‑веса сохранятся в `outputs/my_style/`.

## Инференс
```bash
python src/infer.py \
  --lora_path outputs/my_style \
  --prompt "a portrait in <mystyle> style, dramatic lighting" \
  --num_images 4
```
Изображения сохраняются в `outputs/gen/` (или в путь из `--output_dir`). 