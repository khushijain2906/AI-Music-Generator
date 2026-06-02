# AI Music Generator — LSTM-RNN Pipeline

Generate original MIDI music using a two-layer LSTM trained on any MIDI dataset.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add MIDI files to the midi_data/ folder
mkdir midi_data
# Download from Maestro / Lakh / Bach links below and drop .mid files here

# 3. Run the full pipeline
python music_generator.py
```

Output: `generated_music.mid` + saved model in `checkpoints/`

---

## Pipeline Overview

| Step | What happens |
|------|-------------|
| 1. Data Collection | Scans `midi_data/` for `.mid` files |
| 2. Preprocessing | music21 extracts note/chord tokens → integer sequences |
| 3. Model | 2-layer LSTM (512 units) + Dense + Softmax |
| 4. Training | Adam optimizer, EarlyStopping, ReduceLROnPlateau |
| 5. Generate | Autoregressive sampling with temperature control |
| 6. Export | pretty_midi writes final `.mid` file |

---

## Key Parameters (edit in `__main__` block)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `midi_dir` | `"midi_data"` | Folder containing training MIDI files |
| `sequence_length` | `100` | Context window (notes) fed to the LSTM |
| `epochs` | `150` | Maximum training epochs |
| `batch_size` | `128` | Samples per gradient update |
| `n_generate` | `200` | Number of notes to generate |
| `temperature` | `0.8` | Creativity: 0.5 = safe, 1.0 = balanced, 1.5 = wild |
| `output_midi` | `"generated_music.mid"` | Output file path |

---

## Generate Without Re-training

If you already have a trained checkpoint:

```python
from music_generator import generate_only

generate_only(
    checkpoint="checkpoints/music_model.h5",
    vocab_path="checkpoints/vocabulary.pkl",
    n_generate=300,
    temperature=1.0,
    output_midi="new_piece.mid",
)
```

---

## Free MIDI Datasets

| Dataset | Genre | Size | URL |
|---------|-------|------|-----|
| Maestro v3 | Classical Piano | 200h | magenta.tensorflow.org/datasets/maestro |
| Lakh MIDI | Mixed | 176K files | colinraffel.com/projects/lmd |
| JSBach Chorales | Baroque | 382 pieces | jsbach.net/midi |
| Classical Archives | Classical | Large | classicalarchives.com |

---

## Convert MIDI to Audio

```bash
# Using FluidSynth (needs a soundfont .sf2 file)
fluidsynth -ni soundfont.sf2 generated_music.mid -F output.wav -r 44100

# Using TiMidity++
timidity generated_music.mid -Ow -o output.wav
```

---

## Model Architecture

```
Input (seq_len, 1)
    ↓
LSTM(512, return_sequences=True)
    ↓
Dropout(0.3)
    ↓
LSTM(512)
    ↓
Dropout(0.3)
    ↓
Dense(256, relu)
    ↓
BatchNormalization
    ↓
Dense(vocab_size, softmax)   ← predicts next note
```
