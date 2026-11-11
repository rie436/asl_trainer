# ASL Fingerspelling Trainer

A webcam-based tool that recognizes ASL alphabet signs and quizzes you
to help you learn them. Built with Python, MediaPipe, and scikit-learn.

## How it works

1. **MediaPipe** finds 21 hand landmarks in each webcam frame.
2. Landmarks are normalized relative to the wrist.
3. A **Random Forest** classifier predicts the letter.
4. A practice mode quizzes you and scores correct signs.

## Setup (Windows, Python 3.10, 64-bit)

    py -3.10 -m venv .venv
    .venv\Scripts\activate
    python -m pip install -r requirements.txt

## Usage

    python 1_test_camera.py          # check camera + hand tracking
    python 2_collect_data.py A       # record samples for letter A
    python 3_train.py                # train the model
    python 4_recognize.py            # live recognition
    python 5_practice.py             # practice quiz

## Current letters

A, B, C, L, Y

## Limitations

- Static signs only (J and Z need motion).
- Trained on one person's hand, so accuracy may vary for others.
- Fingerspelling is only a small part of ASL.

## Ideas for next steps

- Add more letters
- Show reference images as hints
- Track weak letters
- Word-level recognition with sequence models
