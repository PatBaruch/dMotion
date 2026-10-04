# Teach dMotion to recognize displayed cash

The original detector guesses from words such as "paper money." Your custom
detector learns from reviewed examples of fans, stacks, and single bills. The
internal class name remains `money_spread` for compatibility. The camera, white
box, and sound stay the same; trained mode changes the model that recognizes cash.

The pipeline is:

**Collect photos → review boxes → split sessions → train → test new examples.**

Downloads, frame extraction, dataset formatting, and training are automated.
Reviewing the labels is the essential human step. A bad box teaches the model the
wrong thing, even if you downloaded hundreds of pictures.

## Use AI to suggest boxes from videos

Import local videos, then run automatic labeling before opening the review page:

```sh
.venv/bin/dmotion import data/videos/first.mp4 data/videos/second.mp4 data/videos/third.mp4 --interval 1
make setup-labeling  # One-time optional dependency setup, after make setup.
make auto-label
make label
```

You can also double-click `auto-label.command`. This uses Grounding DINO with
banknote prompts to propose cash boxes for unreviewed pictures. The first run
downloads roughly 689 MB of weights into the project cache. Your video frames
are processed locally. Importing the same unchanged videos again does not
duplicate the frames. Reviewed pictures are never relabeled automatically.

The review page opens with AI boxes already drawn. Click **Save cash + next** to
accept a correct proposal; correct any incomplete or incorrect boxes first.
Frames where the AI found nothing still need checking for missed cash. Only
explicitly accepted labels enter training. Suggestions, confidence scores, and
model details remain separate from reviewed boxes. Preview sheets and a report
are saved in `outputs/video-autolabel/`.

For the broader cash-display task, fans, stacks, and single bills are positives.
Cards, receipts, phones, and empty hands are negatives. Keep each box around the
cash rather than the whole person. The internal class name remains
`money_spread` so the existing training and camera commands stay compatible.

The existing prompt detector is also available as a lighter alternative:

```sh
.venv/bin/dmotion auto-label --engine world --prompt "paper money" --confidence 0.1
```

Automatic labels can contain errors. Review every frame you want to include in
training, and retain independently reviewed test videos to measure missed cash
and false alarms. Finishing auto-labeling does not train or replace the live model.

Training stops if any AI suggestions still await review. Accept or correct their
cash boxes, mark frames with no cash as **No cash**, or skip unsuitable frames.
Then click **Finish labeling** and run `make train`.

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

For a positive session, hold and move a money fan, stack, or single bill: turn it
slightly, vary the distance, and show different parts of the frame. Record another
session with a different background or lighting. Avoid collecting a long sequence
of almost identical frames.

For a negative session, show empty hands, cards, receipts, phones, and everyday
objects without banknotes. These examples teach the model when it should stay
silent. Single bills and closed stacks of notes remain positive cash examples.

```sh
.venv/bin/dmotion collect --seconds 20 --interval 1 --kind positive
.venv/bin/dmotion collect --seconds 20 --interval 1 --kind negative
```

`--kind` describes your intention for the session. **It does not label the saved
photos.** Every captured photo starts unreviewed, since a positive recording can
also contain frames where the cash is out of view.

Press Q or Escape, or close the camera window, to finish collection early. Each
new recording gets its own session group automatically.

You can import existing photos, a folder, or a video:

```sh
.venv/bin/dmotion import data/my-photos --group living-room-evening
.venv/bin/dmotion import data/videos/my-recording.mp4 --interval 1
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

- For a money fan or stack, draw one tight rectangle around the **entire visible
  cash bundle**. Include all its banknotes in that box.
- For a single bill, draw one tight rectangle around the bill.
- For separate cash bundles or bills, draw separate rectangles.
- Keep the box around the cash. Do not include the whole person or extra background.
- If no cash is visible, save the photo as a negative example with no boxes.
- Skip pictures that are too blurry, irrelevant, or ambiguous to label confidently.

| Control | What it does |
| --- | --- |
| Save cash + next | Save your drawn boxes as a positive example and move on |
| No cash + next | Explicitly save a negative example with no boxes |
| Skip + next | Exclude this picture from training |
| Undo box / Clear boxes | Remove the last box or all boxes so you can draw again |
| Previous / Next | Revisit saved pictures and edit their labels |
| Next unreviewed | Find a picture that still needs labeling |
| Finish labeling | Stop the local server when you are done |

Save a picture before moving on or finishing: drawing and clearing boxes changes
the preview, while **Save cash + next** keeps the edited label. To replace a
starting box, clear the boxes, draw a better rectangle, and save it. Shortcuts are
Enter for cash, 0 for no cash, arrow keys to browse, and Delete to clear.

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

Record and review at least **three independent groups containing cash**.
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

To give a small dataset more learning steps, you can run
`.venv/bin/dmotion train --epochs 100 --patience 30`. `--patience` controls how
many epochs without improved validation results are allowed before stopping;
the default is 10, and 0 disables early stopping. The selected value is saved
in the training report.

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
.venv/bin/dmotion image data/photos/new-spread.jpg --mode trained --show
```

Or double-click `trained.command`. Use new photos and new webcam conditions:
show a fan, stack, and single bill, then remove the cash and show cards and empty
hands. The alert should happen for cash and stay quiet without it. The existing
sound confirmation and cooldown settings still apply.

Training completion does not establish accuracy. Good scores on a few held-out
images can be misleading. If the app misses real cash or triggers on cards,
collect those examples, label them, add independent sessions, and train again.
Keep some new sessions out of training to check the improvement honestly.

In the report, precision describes how often the model's predicted boxes are
correct, recall describes how much real cash it finds, and mAP summarizes box
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

Provided originals belong in `data/videos/` and `data/photos/`; reviewed frame
copies stay in `data/training/`. Run `make organize-media` to sort loose uploads.
See [file layout](FILE_LAYOUT.md). To understand the distinct fine-tuning route
for the pretrained reference model, read [YOLOE fine-tuning](YOLOE_FINETUNING.md).
