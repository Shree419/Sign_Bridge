# The `data/` folder

Everything in here is created on your computer and is **not** committed to git
(see `.gitignore`). Datasets are large, and recordings of people signing are personal data.

```text
data/
├── raw/
│   └── asl-signs/                ← put the downloaded Kaggle dataset here
│       ├── train.csv
│       └── train_landmark_files/<participant_id>/<sequence_id>.parquet
└── samples/                      ← created automatically
    ├── index.csv                 ← one row per sample (the table of contents)
    ├── islr/<sample_id>.npy      ← samples converted from the Kaggle dataset
    ├── custom/<sample_id>.npy    ← samples you recorded with collect_samples.py
    └── app_corrections/<id>.npy  ← samples saved from the app's "Correct a mistake" panel
```

## Getting the Kaggle "asl-signs" dataset

1. Create a free account on [kaggle.com](https://www.kaggle.com).
2. Open the competition page,
   [Google – Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/data),
   and accept its rules. Kaggle will not let you download the data until you do.
   The data is licensed under **CC BY 4.0**, so you may use it as long as you
   credit its creators (the Deaf Professional Arts Network and the Georgia
   Institute of Technology). See "Dataset license and attribution" in the main README.
3. Download the data in one of two ways:
   - **In the browser:** click "Download All", then unzip it into `data/raw/asl-signs/`.
   - **With the Kaggle command-line tool:** first put your API token in
     `~/.kaggle/kaggle.json` (see Kaggle → Settings → API), then run:
     ```bash
     pip install kaggle
     kaggle competitions download -c asl-signs -p data/raw/asl-signs
     ```
     Then unzip the downloaded file in that folder.
4. Convert the signs listed in `config.yaml` into project samples:
   ```bash
   python -m training.prepare_islr
   ```
   The full download is very large (tens of GB). The converter reads only the
   clips for the signs you selected. Use `--max-per-sign 100` for a quick first try.

If you cannot download the dataset, you can still build a small model from your
own recordings (see the main README, "Recording your own training data").

## The sample format

### `index.csv`

| column       | meaning                                                                 |
|--------------|-------------------------------------------------------------------------|
| `sample_id`  | unique name of the sample                                               |
| `label`      | which sign it shows, e.g. `hello` (this is the **label** the model learns) |
| `signer_id`  | who performed it. Used to keep each person in only one of train/validation/test |
| `source`     | `islr`, `custom` or `app_corrections`                                   |
| `file_path`  | path of the `.npy` file, relative to `data/samples/`                    |
| `num_frames` | how many video frames the sample has                                    |
| `created_at` | when it was recorded (empty for imported samples)                       |

### Each `.npy` file

A NumPy array of shape **`(num_frames, 2, 21, 3)`**, stored as float32:

- `num_frames`: one entry per video frame.
- `2`: hand slots. Slot 0 is the hand the detector labelled **left** and slot 1 is the hand it labelled **right**.
- `21`: MediaPipe's hand landmarks (0 = wrist, 4 = thumb tip, 8 = index tip, ... 20 = little-finger tip).
- `3`: `x`, `y` (fractions of the image width/height, 0–1) and `z` (depth relative to the wrist).

A hand that was not detected in a frame is stored as `NaN`. You can open a file with:

```python
import numpy as np
sequence = np.load("data/samples/custom/<sample_id>.npy")
print(sequence.shape)
```

Only landmark numbers are stored. No images or video are saved, unless you
explicitly run the collector with `--save-video` (those videos go to
`data/samples/custom_videos/`).
