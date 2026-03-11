"""Download LibriSpeech test-clean and select 200 utterances for calibration."""
import os
import numpy as np
import soundfile as sf

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
N_FILES = 200


def download_and_select():
    """Download LibriSpeech test-clean via HuggingFace datasets, select 200 files."""
    from datasets import load_dataset

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading LibriSpeech test-clean from HuggingFace...")
    ds = load_dataset("librispeech_asr", "clean", split="test", trust_remote_code=True)

    # Group by speaker for diversity
    speaker_to_indices = {}
    for idx, item in enumerate(ds):
        spk = item["speaker_id"]
        if spk not in speaker_to_indices:
            speaker_to_indices[spk] = []
        speaker_to_indices[spk].append(idx)

    speakers = sorted(speaker_to_indices.keys())
    print(f"Found {len(speakers)} speakers, {len(ds)} total utterances")

    # Round-robin selection across speakers
    selected = []
    speaker_cycle = 0
    while len(selected) < N_FILES:
        spk = speakers[speaker_cycle % len(speakers)]
        indices = speaker_to_indices[spk]
        # Pick next un-selected index from this speaker
        pick_idx = len([s for s in selected if ds[s]["speaker_id"] == spk])
        if pick_idx < len(indices):
            selected.append(indices[pick_idx])
        speaker_cycle += 1
        # Safety: break if we've cycled through all speakers without finding new items
        if speaker_cycle > N_FILES * len(speakers):
            break

    selected = selected[:N_FILES]
    print(f"Selected {len(selected)} utterances from {len(set(ds[i]['speaker_id'] for i in selected))} speakers")

    # Save as FLAC
    for i, idx in enumerate(selected):
        item = ds[idx]
        audio = np.array(item["audio"]["array"], dtype=np.float32)
        sr = item["audio"]["sampling_rate"]
        spk = item["speaker_id"]
        out_path = os.path.join(OUTPUT_DIR, f"spk{spk}_{i:04d}.flac")
        sf.write(out_path, audio, sr)

    print(f"Saved {len(selected)} files to {OUTPUT_DIR}")

    # Stats
    spk_counts = {}
    for idx in selected:
        spk = ds[idx]["speaker_id"]
        spk_counts[spk] = spk_counts.get(spk, 0) + 1
    counts = list(spk_counts.values())
    print(f"Speaker diversity: {len(spk_counts)} speakers, "
          f"min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f} files/speaker")


if __name__ == "__main__":
    download_and_select()
