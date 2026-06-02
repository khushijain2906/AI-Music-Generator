"""
AI Music Generation with LSTM (RNN)
Full pipeline: Data Collection → Preprocessing → Model → Training → MIDI Export
Requirements: pip install music21 tensorflow numpy pretty_midi
"""
import streamlit as st
import os
import pickle
import numpy as np
from glob import glob

# ─────────────────────────────────────────────────────────
# 1. DATA COLLECTION & PREPROCESSING  (music21)
# ─────────────────────────────────────────────────────────

def collect_midi_files(directory: str = "midi_data", extensions=("*.mid", "*.midi")):
    """Scan a directory for MIDI files."""
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(directory, "**", ext), recursive=True))
    print(f"[Data] Found {len(files)} MIDI files in '{directory}'")
    return files


def extract_notes_from_midi(midi_path: str):
    """
    Parse a MIDI file with music21 and extract note/chord tokens.
    Returns a list of strings like 'C4', 'E4.G4.C5', '0.5' (rest).
    """
    from music21 import converter, instrument, note, chord, stream

    try:
        score = converter.parse(midi_path)
    except Exception as e:
        print(f"  [!] Skipping {midi_path}: {e}")
        return []

    notes_out = []
    parts = instrument.partitionByInstrument(score)
    target = parts.parts[0] if parts else score.flatten()
    for element in target.flatten().notes:
        if isinstance(element, note.Note):
            notes_out.append(str(element.pitch))          # e.g. "C4"
        elif isinstance(element, chord.Chord):
            # Encode chord as dot-separated pitches
            notes_out.append(".".join(str(n) for n in element.normalOrder))
    return notes_out


def build_dataset(midi_files: list, sequence_length: int = 100):
    """
    Extract notes from all MIDI files and build integer-encoded sequences.
    Returns:
        X            – input sequences  (N, seq_len, 1)  float32
        y            – one-hot targets  (N, vocab_size)  float32
        note_to_int  – mapping dict
        int_to_note  – reverse mapping dict
        n_vocab      – vocabulary size
    """
    import tensorflow as tf

    all_notes = []
    for f in midi_files:
        all_notes.extend(extract_notes_from_midi(f))

    if not all_notes:
        raise ValueError("No notes found. Check your midi_data/ directory.")

    print(f"[Data] Total notes extracted: {len(all_notes)}")

    vocab = sorted(set(all_notes))
    n_vocab = len(vocab)
    note_to_int = {n: i for i, n in enumerate(vocab)}
    int_to_note = {i: n for n, i in note_to_int.items()}

    print(f"[Data] Vocabulary size: {n_vocab}")

    # Encode all notes as integers
    encoded = [note_to_int[n] for n in all_notes]

    X_raw, y_raw = [], []
    for i in range(len(encoded) - sequence_length):
        X_raw.append(encoded[i : i + sequence_length])
        y_raw.append(encoded[i + sequence_length])

    n_samples = len(X_raw)
    print(f"[Data] Training samples: {n_samples}")

    # Normalize & reshape
    X = np.reshape(X_raw, (n_samples, sequence_length, 1)) / float(n_vocab)
    y = tf.keras.utils.to_categorical(y_raw, num_classes=n_vocab)

    return X.astype("float32"), y.astype("float32"), note_to_int, int_to_note, n_vocab


# ─────────────────────────────────────────────────────────
# 2. MODEL ARCHITECTURE  (LSTM-RNN)
# ─────────────────────────────────────────────────────────

def build_lstm_model(sequence_length: int, n_vocab: int):
    """
    Two-layer LSTM with Dropout, BatchNorm, and a softmax output.
    Architecture:
        LSTM(512, return_sequences=True) → Dropout(0.3)
        LSTM(512)                        → Dropout(0.3)
        Dense(256, relu)                 → BatchNorm
        Dense(n_vocab, softmax)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input

    inputs = Input(shape=(sequence_length, 1), name="note_sequence")

    x = layers.LSTM(512, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(0.3, name="drop_1")(x)

    x = layers.LSTM(512, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(0.3, name="drop_2")(x)

    x = layers.Dense(256, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)

    outputs = layers.Dense(n_vocab, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="MusicLSTM")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────────────────────

def train_model(
    model,
    X,
    y,
    epochs: int = 150,
    batch_size: int = 128,
    checkpoint_path: str = "checkpoints/music_model.h5",
):
    """Train with ModelCheckpoint + EarlyStopping + LR reduction."""
    import tensorflow as tf

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\n[Train] Starting training for up to {epochs} epochs …")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_split=0.1,
        shuffle=True,
    )
    return history


# ─────────────────────────────────────────────────────────
# 4. MUSIC GENERATION
# ─────────────────────────────────────────────────────────

def generate_notes(
    model,
    seed_sequence: list,
    int_to_note: dict,
    n_vocab: int,
    n_generate: int = 200,
    temperature: float = 1.0,
    sequence_length: int = 100,
):
    """
    Autoregressively generate n_generate notes.
    temperature > 1  → more random / creative
    temperature < 1  → more conservative / repetitive
    """
    pattern = list(seed_sequence)
    generated = []

    for _ in range(n_generate):
        net_in = np.reshape(pattern, (1, sequence_length, 1)) / float(n_vocab)
        pred = model.predict(net_in, verbose=0)[0].astype("float64")

        # Temperature scaling
        pred = np.log(pred + 1e-8) / temperature
        pred = np.exp(pred) / np.sum(np.exp(pred))

        idx = np.random.choice(len(pred), p=pred)
        generated.append(int_to_note[idx])
        pattern.append(idx)
        pattern = pattern[1:]   # slide the window

    return generated


# ─────────────────────────────────────────────────────────
# 5. MIDI EXPORT  (pretty_midi + music21 fallback)
# ─────────────────────────────────────────────────────────

def notes_to_midi(
    generated_notes: list,
    output_path: str = "generated_music.mid",
    tempo_bpm: int = 120,
    note_duration: float = 0.5,
):
    """
    Convert generated note tokens back to a MIDI file.
    Supports single pitches ('C4') and chords ('0.3.7').
    """
    try:
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
        piano = pretty_midi.Instrument(program=0, name="Piano")
        seconds_per_step = 60.0 / tempo_bpm * note_duration

        current_start = 0.0
        for token in generated_notes:
            if "." in token:
                # Chord: decode note numbers
                pitches = [int(p) + 60 for p in token.split(".")]
            else:
                # Single note name → MIDI number via music21
                try:
                    from music21 import pitch as m21pitch
                    pitches = [m21pitch.Pitch(token).midi]
                except Exception:
                    pitches = [60]  # fallback to middle C

            end_time = current_start + seconds_per_step
            for p in pitches:
                p = max(0, min(127, p))
                piano.notes.append(
                    pretty_midi.Note(
                        velocity=90,
                        pitch=p,
                        start=current_start,
                        end=end_time,
                    )
                )
            current_start = end_time

        midi.instruments.append(piano)
        midi.write(output_path)
        print(f"[Export] MIDI saved → {output_path}")

    except ImportError:
        # Fallback: music21 stream writer
        from music21 import stream, note, chord as m21chord, tempo as m21tempo

        out_stream = stream.Stream()
        out_stream.append(m21tempo.MetronomeMark(number=tempo_bpm))

        for token in generated_notes:
            dur = note_duration
            if "." in token:
                pitches = [int(p) + 60 for p in token.split(".")]
                c = m21chord.Chord(pitches)
                c.duration.quarterLength = dur
                out_stream.append(c)
            else:
                n = note.Note(token)
                n.duration.quarterLength = dur
                out_stream.append(n)

        out_stream.write("midi", fp=output_path)
        print(f"[Export] MIDI saved → {output_path}  (via music21 fallback)")

    return output_path


# ─────────────────────────────────────────────────────────
# 6. FULL PIPELINE  (main entry point)
# ─────────────────────────────────────────────────────────

def run_pipeline(
    midi_dir: str = "midi_data",
    sequence_length: int = 100,
    epochs: int = 150,
    batch_size: int = 128,
    n_generate: int = 200,
    temperature: float = 0.8,
    output_midi: str = "generated_music.mid",
    checkpoint: str = "checkpoints/music_model.h5",
):
    print("=" * 60)
    print("  AI Music Generator  –  LSTM-RNN Pipeline")
    print("=" * 60)

    # ── Step 1: Collect MIDI files ──────────────────────────
    midi_files = collect_midi_files(midi_dir)
    if not midi_files:
        print(f"\n[!] No MIDI files found in '{midi_dir}'.")
        print("    Download a dataset and place .mid files there, e.g.:")
        print("    • Maestro:  https://magenta.tensorflow.org/datasets/maestro")
        print("    • Lakh:     https://colinraffel.com/projects/lmd/")
        print("    • Bach:     https://www.jsbach.net/midi/\n")
        return

    # ── Step 2: Preprocess ──────────────────────────────────
    X, y, note_to_int, int_to_note, n_vocab = build_dataset(
        midi_files, sequence_length=sequence_length
    )

    # Save vocabulary for later use
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/vocabulary.pkl", "wb") as f:
        pickle.dump({"note_to_int": note_to_int, "int_to_note": int_to_note, "n_vocab": n_vocab}, f)
    print("[Data] Vocabulary saved → checkpoints/vocabulary.pkl")

    # ── Step 3: Build model ─────────────────────────────────
    model = build_lstm_model(sequence_length, n_vocab)
    model.summary()

    # ── Step 4: Train ───────────────────────────────────────
    # If a checkpoint exists, load weights and fine-tune
    if os.path.exists(checkpoint):
        print(f"[Train] Loading existing checkpoint: {checkpoint}")
        model.load_weights(checkpoint)
    else:
        train_model(model, X, y, epochs=epochs, batch_size=batch_size,
                    checkpoint_path=checkpoint)

    # ── Step 5: Generate ────────────────────────────────────
    print(f"\n[Generate] Sampling {n_generate} notes (temperature={temperature}) …")

    # Pick a random seed sequence from training data
    seed_idx = np.random.randint(0, len(X) - 1)
    seed = (X[seed_idx].flatten() * n_vocab).astype(int).tolist()

    generated = generate_notes(
        model, seed, int_to_note, n_vocab,
        n_generate=n_generate,
        temperature=temperature,
        sequence_length=sequence_length,
    )
    print(f"[Generate] First 10 tokens: {generated[:10]}")

    # ── Step 6: Export MIDI ─────────────────────────────────
    notes_to_midi(generated, output_path=output_midi)

    print("\n[Done] Pipeline complete!")
    print(f"  MIDI  →  {output_midi}")
    print(f"  Model →  {checkpoint}")


# ─────────────────────────────────────────────────────────
# GENERATE-ONLY MODE  (if model already trained)
# ─────────────────────────────────────────────────────────

def generate_only(
    checkpoint: str = "checkpoints/music_model.h5",
    vocab_path: str = "checkpoints/vocabulary.pkl",
    sequence_length: int = 100,
    n_generate: int = 200,
    temperature: float = 0.8,
    output_midi: str = "generated_music.mid",
):
    """Load a pre-trained model and generate music without re-training."""
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    note_to_int = vocab["note_to_int"]
    int_to_note = vocab["int_to_note"]
    n_vocab      = vocab["n_vocab"]

    model = build_lstm_model(sequence_length, n_vocab)
    model.load_weights(checkpoint)
    print(f"[Generate] Model loaded from {checkpoint}")

    seed = np.random.randint(0, n_vocab, size=sequence_length).tolist()
    generated = generate_notes(
        model, seed, int_to_note, n_vocab,
        n_generate=n_generate,
        temperature=temperature,
        sequence_length=sequence_length,
    )
    notes_to_midi(generated, output_path=output_midi)
    return output_midi


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    st.title("🎵 AI Music Generator")
    st.write("Generate music using LSTM-RNN")

    if st.button("🎼 Generate Music"):

        with st.spinner("Training and generating music..."):

            run_pipeline(
                midi_dir="midi_data",
                sequence_length=100,
                epochs=150,
                batch_size=128,
                n_generate=200,
                temperature=0.8,
                output_midi="generated_music.mid",
                checkpoint="checkpoints/music_model.h5",
            )

        st.success("Music Generated Successfully!")

        with open("generated_music.mid", "rb") as f:
            st.download_button(
                "⬇ Download MIDI",
                f,
                file_name="generated_music.mid",
                mime="audio/midi"
            )