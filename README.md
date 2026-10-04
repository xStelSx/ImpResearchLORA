# FLUX.2 Klein — LoRA / AdaLoRA / IA3 Fine-Tuning Benchmark

Ноутбук для тонкой настройки (fine-tuning) диффузионной модели **FLUX.2 Klein (4B)** тремя PEFT-методами — **LoRA**, **AdaLoRA** и **IA3**, с последующей генерацией изображений и оценкой их соответствия промптам через **CLIP**.

Проект рассчитан на запуск в **Google Colab** на GPU **NVIDIA T4** и оптимизирован так, чтобы уложиться в ~13 GB VRAM.

---

## Содержание

- [Возможности](#возможности)
- [Требования](#требования)
- [Секреты Colab](#секреты-colab)
- [Структура пайплайна](#структура-пайплайна)
- [Подготовка данных](#подготовка-данных)
- [Ключевые параметры](#ключевые-параметры)
- [PEFT-конфигурации](#peft-конфигурации)
- [Промпты для генерации](#промпты-для-генерации)
- [Как запускать](#как-запускать)
- [Что выводится в консоль](#что-выводится-в-консоль)
- [Структура выходных файлов](#структура-выходных-файлов)
- [Пример итоговой таблицы CLIP](#пример-итоговой-таблицы-clip)
- [Возможные проблемы](#возможные-проблемы)
- [Использованные источники](#использованные-источники)
- [Лицензия и назначение](#лицензия-и-назначение)

---

## Возможности

- **Три PEFT-метода в одном прогоне**: LoRA, AdaLoRA, IA3 — обучаются последовательно на одном и том же датасете, что позволяет честно их сравнить.
- **Экономия VRAM**:
  - Предвычисление VAE-латентов для всех изображений датасета — VAE выгружается из GPU до начала обучения.
  - Предвычисление текстовых эмбеддингов (Qwen3, слои **9 / 18 / 27**, конкатенация → 7680-мерные векторы) — текстовый энкодер выгружается до начала обучения.
  - 8-битная квантизация текстового энкодера через `bitsandbytes`.
  - `enable_gradient_checkpointing()` для трансформера.
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Корректный 4D RoPE** для FLUX.2: отдельные `img_ids` (t, row, col, extra) и `txt_ids` (все нули).
- **Flow Matching** лосс: `target = noise - latents`, MSE между предсказанием модели и таргетом.
- **CLIP-оценка**: `ViT-B/32` считает cosine similarity между сгенерированными изображениями и промптами.
- **Визуализация**: сводная сетка сравнения `comparison_grid.png` (строки — методы, столбцы — промпты, в заголовках — CLIP-score).
- **Автоматический fallback при OOM**: если генерация 512×512 падает, выполняется повтор на 384×384.
- **Интеграция с Google Drive** — адаптеры и картинки сохраняются сразу в Drive и переживают перезапуск рантайма.

---

## Требования

| Компонент | Значение |
|---|---|
| Среда | Google Colab |
| GPU | NVIDIA **T4** (минимум ~13 GB свободной VRAM) |
| Python | 3.10+ |
| Основные библиотеки | `diffusers` (из git), `peft` (из git), `transformers`, `accelerate`, `datasets`, `bitsandbytes`, `sentencepiece`, `torchao>=0.17.0`, `CLIP` (OpenAI, из git) |
| Модель | `black-forest-labs/FLUX.2-klein-4B` |

### Установка зависимостей

Выполняется в первой ячейке ноутбука:

```bash
!pip uninstall -y diffusers peft transformers accelerate bitsandbytes xformers triton -q
!pip install git+https://github.com/huggingface/diffusers.git -q
!pip install git+https://github.com/huggingface/peft.git -q
!pip install transformers accelerate datasets bitsandbytes sentencepiece -q
!pip install --no-deps --upgrade "torchao>=0.17.0" -q
!pip install git+https://github.com/openai/CLIP.git -q
```

>  После установки `diffusers` и `peft` из git **обязательно перезапустите рантайм** (`Runtime → Restart runtime`), иначе возможны конфликты версий.

---

## Секреты Colab

Перед запуском добавьте в **Colab → Secrets** (значок 🔑 в левой панели):

| Имя | Назначение |
|---|---|
| `HF_TOKEN` | Токен Hugging Face. Нужен для скачивания FLUX.2 Klein и доступа к текстовому энкодеру Qwen3. |

Получить токен: https://huggingface.co/settings/tokens

Если `HF_TOKEN` не задан, ноутбук напечатает `HF_TOKEN не найден` и продолжит работу — но скачивание модели может упасть.

---

## Структура пайплайна

```
┌────────────────────────────────────────────────────┐
│ 0. Установка зависимостей                          │
│ 1. Проверка свободной VRAM                         │
│ 2. Логин в Hugging Face + монтирование Drive       │
│ 3. Загрузка и валидация датасета (metadata.csv)    │
│ 4. Предвычисление VAE-латентов (выгрузка VAE)      │
│ 5. Предвычисление текстовых эмбеддингов (выгрузка) │
│ 6. Patchify + построение 4D ids                    │
│ 7. Конфигурации PEFT (LoRA / AdaLoRA / IA3)        │
│ 8. Обучение каждого метода (по 500 шагов)          │
│ 9. Генерация изображений по 3 промптам             │
│ 10. CLIP-оценка + построение сетки сравнения       │
└────────────────────────────────────────────────────┘
```

---

## Подготовка данных

Ожидается CSV-файл по пути:

```
/content/drive/MyDrive/Colab Notebooks/LoraDataset/metadata.csv
```

Колонки:

| Колонка | Тип | Описание |
|---|---|---|
| `file_name` | str | Путь к изображению. Может быть абсолютным или относительным — во втором случае склеивается с `BASE_DIR`. |
| `caption`   | str | Текстовое описание изображения. |

Нормализация путей:
- `\` заменяется на `/`;
- снимаются кавычки (`"`, `'`);
- относительные пути склеиваются с `BASE_DIR`;
- несуществующие файлы молча отбрасываются.

Пример строки `metadata.csv`:

```csv
file_name,caption
images/flowers.png,"qwerty style, A work by an unknown artist from the 18th century. Oval, depicting a bouquet of flowers on a dark background"
images/tower.png,"qwerty style, A work by an unknown artist from the 18th century. Square, depicting a landscape with a tower on a rocky riverbank"
```

Датасет ограничивается `LIMIT_N = 10` строками (см. [Ключевые параметры](#ключевые-параметры)).

---

## Ключевые параметры

Все параметры задаются в начале основной ячейки ноутбука:

```python
BASE_DIR    = "/content/drive/MyDrive/Colab Notebooks/LoraDataset"
OUTPUT_ROOT = "/content/drive/MyDrive/Colab Notebooks/flux_results"
MODEL_ID    = "black-forest-labs/FLUX.2-klein-4B"

IMAGE_SIZE = 256       # разрешение обучения (латент = 256/8 = 32×32)
MAX_STEPS  = 500       # шагов обучения на каждый метод
LR         = 2e-4      # learning rate для AdamW
GRAD_ACCUM = 4         # шагов накопления градиента
LIMIT_N    = 10        # ограничение размера датасета
```

**Размерности тензоров (при `IMAGE_SIZE=256`)**:

| Тензор | Форма | Комментарий |
|---|---|---|
| VAE-латент | `[1, 8, 32, 32]` | после `vae.encode()` |
| `packed` | `[1, 256, 128]` | после patchify (2×2 патчи, 8·4=32 канала? → 128 после reshape по C·4) |
| `img_ids` | `[1, 256, 4]` | 4D RoPE для изображения |
| `ehs` (encoder hidden) | `[1, 512, 7680]` | конкатенация 3 слоёв Qwen3 (2560·3) |
| `txt_ids` | `[1, 512, 4]` | 4D RoPE для текста (все нули) |

---

## PEFT-конфигурации

Общие `target_modules` для всех методов:

```python
TARGET_MODULES = ["to_q", "to_k", "to_v"]
```

| Метод | Конфигурация |
|---|---|
| **LoRA** | `r=8, lora_alpha=8, lora_dropout=0.1, bias="none"` |
| **AdaLoRA** | `r=8, lora_alpha=8, lora_dropout=0.1, bias="none", total_step=500, init_r=12, target_r=4, beta1=0.85, beta2=0.85, orth_reg_weight=0.5` |
| **IA3** | `target_modules=["to_q","to_k","to_v"], feedforward_modules=[]` |

Методы обучаются по очереди в порядке: `["lora", "adalora", "ia3"]`.

---

## Промпты для генерации

Три заранее заданных промпта в стиле «qwerty style», описывающих работы неизвестного художника XVIII века:

```python
prompts = [
    "qwerty style, A work by an unknown artist from the 18th century. Oval, depicting a bouquet of flowers on a dark background",
    "qwerty style, A work by an unknown artist from the 18th century. Square, depicting a landscape with a tower on a rocky riverbank",
    "qwerty style, A work by an unknown artist from the 18th century. Rectangular, depicting a landscape with a waterfall in the foreground, a hut in the background",
]
```

Параметры генерации:

- `height = width = 512` (fallback на 384×384 при OOM)
- `guidance_scale = 1.0`
- `num_inference_steps = 4` (FLUX.2 Klein — few-step модель)

---

## Как запускать

1. Откройте ноутбук в **Google Colab**.
2. Выберите среду выполнения: **Runtime → Change runtime type → T4 GPU**.
3. Убедитесь, что `HF_TOKEN` добавлен в **Secrets** (🔑).
4. Выполните первую ячейку с `pip install` (см. [Установка зависимостей](#установка-зависимостей)).
5. **Перезапустите рантайм** (`Runtime → Restart runtime`).
6. Запустите основную (большую) ячейку целиком.
7. Дождитесь завершения — все результаты окажутся в `/content/drive/MyDrive/Colab Notebooks/flux_results/`.

**Ориентировочное время на T4**:

| Этап | Время |
|---|---|
| Установка зависимостей | ~100 сек |
| Скачивание FLUX.2 Klein | ~1–3 мин (кэшируется между запусками) |
| Предвычисление латентов | < 1 мин для 10 изображений |
| Предвычисление эмбеддингов | ~1–2 мин |
| Обучение (3 × 500 шагов) | ~1.5–2.5 ч |
| Генерация (3 метода × 3 промпта) | ~5–15 мин |
| CLIP + сетка | < 1 мин |

---

## Что выводится в консоль

### Начало

```
VRAM: free=15.xx GB / total=15.xx GB
 Логин в HF
Mounted at /content/drive
Используем 10 изображений

1. Предвычисление латентов...
   латентов: 10, форма: torch.Size([1, 8, 32, 32])

2. Предвычисление эмбеддингов...
   эмбеддингов: 10, форма: torch.Size([1, 512, 7680])

 patchify: torch.Size([1, 8, 32, 32]) → torch.Size([1, 256, 128])
 img_ids:  torch.Size([1, 256, 4])
```

### Обучение (для каждого метода)

```
============================================================
 LORA
============================================================
trainable params: 147,456 || all params: 4,000,000,000 || trainable%: 0.0037
   step    0, loss: 0.9532
   step   20, loss: 0.8471
   step   40, loss: 0.7914
   ...
 lora сохранён: /content/drive/.../lora_results
```

### Генерация

```
============================================================
ГЕНЕРАЦИЯ
============================================================

=== lora ===
   saved lora 0
   saved lora 1
   saved lora 2

=== adalora ===
   saved adalora 0
   ...
```

### CLIP

```
============================================================
CLIP
============================================================
Найдены методы: ['adalora', 'ia3', 'lora']

CLIP Scores:
Method    | P0 | P1 | P2 | Avg
adalora   | 0.2712 | 0.2534 | 0.2601 | 0.2616
ia3       | 0.2447 | 0.2318 | 0.2503 | 0.2423
lora      | 0.2831 | 0.2709 | 0.2764 | 0.2768

 Сетка: /content/drive/.../comparison_grid.png

 DONE: /content/drive/MyDrive/Colab Notebooks/flux_results
```

---

## Структура выходных файлов

```
/content/drive/MyDrive/Colab Notebooks/flux_results/
├── lora_results/                    # PEFT-адаптер LoRA
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── adalora_results/                 # PEFT-адаптер AdaLoRA
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── ia3_results/                     # PEFT-адаптер IA3
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── lora_result_0.png                # сгенерированные изображения
├── lora_result_1.png
├── lora_result_2.png
├── adalora_result_0.png
├── adalora_result_1.png
├── adalora_result_2.png
├── ia3_result_0.png
├── ia3_result_1.png
├── ia3_result_2.png
└── comparison_grid.png              # итоговая сетка: методы × промпты + CLIP
```

---

## Пример итоговой таблицы CLIP

| Method  | P0      | P1      | P2      | Avg    |
|---------|---------|---------|---------|--------|
| adalora | 0.2712  | 0.2534  | 0.2601  | 0.2616 |
| ia3     | 0.2447  | 0.2318  | 0.2503  | 0.2423 |
| lora    | 0.2831  | 0.2709  | 0.2764  | **0.2768** |

**P0 / P1 / P2** — три промпта (цветы, башня, водопад).  
**Avg** — среднее по трём промптам. Чем выше — тем лучше соответствие изображения тексту.

---

## Возможные проблемы

| Проблема | Причина | Решение |
|---|---|---|
| `Мало VRAM. Runtime → Restart runtime.` | На GPU меньше 13 GB свободной памяти | Runtime → Restart runtime, закройте лишние ячейки, не запускайте ничего параллельно |
| `CUDA out of memory` при генерации | 512×512 не влезает | Ноутбук автоматически повторяет с 384×384. При повторной ошибке уменьшите `MAX_STEPS` или `IMAGE_SIZE` |
| `HF_TOKEN не найден` | Не добавлен секрет | Добавьте `HF_TOKEN` в Colab Secrets |
| `ImportError` для `Flux2KleinPipeline` / `Flux2Transformer2DModel` | Не установлена свежая версия `diffusers` из git | Выполните `pip install git+https://github.com/huggingface/diffusers.git` и **перезапустите рантайм** |
| `assert _p.shape[-1] == 128` падает | VAE не от FLUX.2 или `IMAGE_SIZE` не кратен 16 | Убедитесь, что скачивается `black-forest-labs/FLUX.2-klein-4B`, а `IMAGE_SIZE` кратен 16 |
| Пустой датасет / нет изображений | Неверные пути или кодировка в CSV | Проверьте `metadata.csv` (UTF-8, колонки `file_name`, `caption`), проверьте, что файлы действительно существуют |
| Ошибка `bitsandbytes` при загрузке text encoder | Конфликт версий `bitsandbytes` / CUDA | В ноутбуке уже используется `load_in_8bit=True` — перезапустите рантайм после `pip install` |
| `sentence-transformers requires transformers<6.0.0` | Warning от pip | Безвредно, если `transformers` из установки уже совместим |
| Обучение идёт очень долго | Большой `MAX_STEPS`, `LIMIT_N` или `IMAGE_SIZE` | Уменьшите `MAX_STEPS`, `LIMIT_N`, `IMAGE_SIZE`; проверьте `gradient_checkpointing` |
| Ошибка `peft` про `AdaLoraConfig` `total_step` | Нужно указывать вручную | Уже задано: `total_step=MAX_STEPS` |

---

## Использованные источники

- [FLUX.2 Klein (Hugging Face)](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) — базовая модель
- [PEFT (Hugging Face)](https://github.com/huggingface/peft) — LoRA / AdaLoRA / IA3
- [Diffusers (Hugging Face)](https://github.com/huggingface/diffusers) — `Flux2KleinPipeline`, `FlowMatchEulerDiscreteScheduler`
- [OpenAI CLIP](https://github.com/openai/CLIP) — оценка качества генерации
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) — 8-битная квантизация

---

## Лицензия и назначение

Код носит **исследовательский / учебный** характер: демонстрирует сравнение PEFT-методов (LoRA, AdaLoRA, IA3) на компактной версии FLUX.2 Klein (4B) с минимальным потреблением памяти на потребительском GPU (T4).

При использовании учитывайте лицензии:

- **FLUX.2 Klein** — лицензия Black Forest Labs (может ограничивать коммерческое использование).
- **CLIP** — MIT.
- **PEFT / diffusers / transformers** — Apache 2.0.

Автор не несёт ответственности за использование кода в коммерческих целях без проверки соответствующих лицензий базовых моделей и библиотек.