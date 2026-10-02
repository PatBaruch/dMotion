# Teach dMotion what a money spread looks like

The original detector guesses from words such as "paper money." Your custom
detector learns from examples marked `money_spread`. The camera, white box, and
sound stay the same; trained mode changes the model that recognizes the cash.

The pipeline is:

**Collect photos → review boxes → split sessions → train → test new examples.**

Downloads, frame extraction, dataset formatting, and training are automated.
Reviewing the labels is the essential human step. A bad box teaches the model the
wrong thing, even if you downloaded hundreds of pictures.

## 1. Collect examples

Run these commands in the project folder after `make setup`:

```sh
make fetch
make collect
```

The first command downloads the included starter source list. The second captures
a short webcam session as separate photos. You can also double-click
`collect.command`. Give the app camera permission if macOS asks.

The source list is `examples/money-spread-sources.json`. On this laptop, the starter
dataset contains three Google/Pinterest money-spread photos, your euro-spread
photo, and two negative photos. Their starting labels have been reviewed, and you
can inspect and change their boxes in the labeler. Six photos are enough to check
the workflow; any starter weights trained from them are an experiment, not a
reliable webcam detector yet. New downloads still start unreviewed.

For a positive session, hold and move a money fan: turn it slightly, vary the
distance, and show different parts of the frame. Record another session with a
different background or lighting. Avoid collecting a long sequence of almost
identical frames.

For a negative session, show empty hands, a single note, a closed stack of notes,
cards, paper, and everyday objects. These examples teach the model when it should
stay silent.

```sh
.venv/bin/dmotion collect --seconds 20 --interval 1 --kind positive
.venv/bin/dmotion collect --seconds 20 --interval 1 --kind negative
```

`--kind` describes your intention for the session. **It does not label the saved
photos.** Every captured photo starts unreviewed, since a positive recording can
also contain frames where the fan is out of view.

Press Q or Escape, or close the camera window, to finish collection early. Each
new recording gets its own session group automatically.

You can import existing photos, a folder, or a video:

```sh
.venv/bin/dmotion import data/my-photos --group living-room-evening
.venv/bin/dmotion import data/my-recording.mp4 --interval 1
```

Video import samples roughly one frame each second. Frames from the same video
share a group. Use the same `--group` for related photos from one recording or
photo session; give independent sessions different names.

Google images help add variety, but webcam examples are especially useful because
they match the camera and euros you want the app to recognize. Keep downloaded
images local; review the original source's terms before redistributing them.

## 2. Draw and review boxes

```sh
make label
```

Or double-click `label.command`. This opens the local labeling page in your
browser. The photos and labels stay in `data/training/` on your laptop.

- For a money spread, draw one tight rectangle around the **entire visible fan**.
  Include all its banknotes, rather than labeling each note separately.
- For two separate money fans, draw two rectangles.
- Keep the box around the cash. Do not include the whole person or extra background.
- If no fan is visible, save the photo as a negative example with no boxes.
- Skip pictures that are too blurry, irrelevant, or ambiguous to label confidently.

| Control | What it does |
| --- | --- |
| Save spread + next | Save your drawn boxes as a positive example and move on |
| No spread + next | Explicitly save a negative example with no boxes |
| Skip + next | Exclude this picture from training |
| Undo box / Clear boxes | Remove the last box or all boxes so you can draw again |
| Previous / Next | Revisit saved pictures and edit their labels |
| Next unreviewed | Find a picture that still needs labeling |
| Finish labeling | Stop the local server when you are done |

Save a picture before moving on or finishing: drawing and clearing boxes changes
the preview, while **Save spread + next** keeps the edited label. To replace a
starting box, clear the boxes, draw a better rectangle, and save it. Shortcuts are
Enter for a spread, 0 for no spread, arrow keys to browse, and Delete to clear.

The page shows the image source, session group, and reviewed counts. It listens
only on your own laptop at `127.0.0.1`; opening it does not upload your photos.

An empty, unreviewed photo is not a negative example. You must explicitly review
it as negative. This prevents unfinished labeling from accidentally teaching the
model that visible cash is background.

Check your progress with:

```sh
make dataset
```

## 3. Build the dataset and train

Record and review at least **three independent groups containing money spreads**.
Also include negative sessions. Three groups are the minimum needed to put
positive examples into training, validation, and testing; a tiny dataset still
produces a weak experiment.

As an initial target, collect roughly 100–200 varied positive photos and a similar
number of negatives. Different people, note combinations, distances, lighting,
and backgrounds matter more than many nearly identical frames. There is no fixed
image count that guarantees good detection.

```sh
.venv/bin/dmotion build-dataset
make train
```

`make train` builds the dataset and starts a training run. You can also double-click
`train.command`. Only reviewed examples are included. The split keeps each group
together, so the model cannot be tested on neighboring frames of its own training
video.

- **Training set:** examples used to adjust the model.
- **Validation set:** separate examples used to compare training checkpoints.
- **Test set:** held-out examples used to check the chosen model at the end.

The app starts from a small pretrained detector, then adjusts it for the single
class `money_spread`. This is called fine-tuning. An epoch is one pass through the
training set. The default is 30 epochs; actual time depends on your laptop and
dataset, so a complete training run may take longer than an hour.

```sh
.venv/bin/dmotion train --epochs 30 --image-size 640 --device auto
# If Apple GPU training fails:
.venv/bin/dmotion train --epochs 30 --image-size 640 --device cpu
```

The first training run may download pretrained weights. After the weights and
images are available, labeling, training, and detection can run offline. The
trained detector is saved as `models/money-spread.pt`.

The run's settings and held-out test results are saved in
`outputs/training/<run-name>/training-info.json`, with a copy beside the trained
model at `models/money-spread.json`. `data/yolo/export-report.json` records which
session went into each split, its reviewed boxes, and small-dataset warnings.

## 4. Test the result

```sh
make trained
# Check a photo that was not used for training:
.venv/bin/dmotion image data/new-spread.jpg --mode trained --show
```

Or double-click `trained.command`. Use new photos and new webcam conditions:
show a spread, remove it, show a single note, show cards, and show empty hands.
The alert should happen only for a spread. The existing sound confirmation and
cooldown settings still apply.

Training completion does not establish accuracy. Good scores on a few held-out
images can be misleading. If the app misses a real spread or triggers on cards,
collect those examples, label them, add independent sessions, and train again.
Keep some new sessions out of training to check the improvement honestly.

In the report, precision describes how often the model's predicted boxes are
correct, recall describes how many real spreads it finds, and mAP summarizes box
quality across confidence levels and overlap requirements. Look at these alongside
real webcam misses and false alerts; a high score on a tiny test set is not enough.

`make diagnose` still checks the original model with common objects, and `make run`
still uses the original money prompts. Trained mode requires the trained weights
file; it cannot create a custom detector by itself.

## Add more download sources

Create a JSON file containing direct image URLs and their original source pages:

```json
[
  {
    "url": "https://example.org/images/money-fan.jpg",
    "source": "https://example.org/original-page",
    "group": "example-photo-session"
  }
]
```

Then run:

```sh
.venv/bin/dmotion fetch data/my-sources.json --limit 20
make label
```

Use the original image URL from the result, rather than the Google results-page
URL. Keep related pictures in one group. A search result is neither a checked
label nor evidence that a photo shows the correct pose; inspect downloaded photos
before including them in training.

All dataset commands accept `--dataset data/another-dataset` if you want a separate
experiment. Keep that same option across collection, labeling, building, and
training. `--config` selects a different app configuration where supported.

## What version control keeps

Git keeps the code, tests, guides, dependency lockfile, and shared settings. Photos,
labels, training output, and model binaries are ignored so personal data and large
files are not accidentally committed. Back up `data/training/` and your successful
weights separately if you want to preserve them.

For implementation details, see the official
[Ultralytics training guide](https://docs.ultralytics.com/modes/train) and
[object detection dataset format](https://docs.ultralytics.com/datasets/detect).
