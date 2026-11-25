# Tetris AI

A falling-block puzzle game in Python that learns to play itself with **deep Q-learning** (PyTorch).
You can play it yourself, or switch on auto-learning and watch a neural network train on 50 games at once.

## Requirements

- Python 3.10+ with Tkinter (included in most Python installs; on Linux: `sudo apt install python3-tk`)
- [PyTorch](https://pytorch.org/) and NumPy

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python tetris-ai.py
```

### Controls

| Key / Button        | Action                                                        |
|---------------------|---------------------------------------------------------------|
| ← / →               | Move piece                                                    |
| ↑                   | Rotate                                                        |
| ↓                   | Drop one row                                                  |
| **Auto Learning**   | Let the AI play and train (loads the newest saved network)    |
| **Manual Learning** | Back to human play                                            |
| **Save Network**    | Save current weights to `weight/GameQNet_<timestamp>.wgt`     |
| **Load Network**    | Pick a `.wgt` file to load                                    |
| **Restart / Exit**  | Restart after game over / quit                                |

The board is also drawn in the terminal with ANSI colors (turn off with `CONSOLE_RENDER = False`).

## How it works

- **State:** the 20×10 board as color ids (0 = empty), flattened to 200 inputs.
- **Network (`GameNet`):** fully connected 200 → 256 → 64 → 32 → 5, one Q-value per action
  (none, left, right, rotate, down).
- **Policy (`Agent.getAction`):** epsilon-greedy. Epsilon decays from 1.0 to 0.05 over ~220 games,
  with extra randomness at the start of each game.
- **Reward (`Agent.getReward`):** large bonuses for clearing lines, penalties for height, holes,
  bumpiness and near-top columns, small bonuses for compact rows near the floor; −40,000 for game over.
- **Training (`GameTrainer.trainStep`):** one online Q-learning update per move,
  target = `reward + γ · max Q(next_state)` with γ = 0.9, Smooth L1 loss, Adam (lr 0.001).
- **Parallel games:** `PARALLEL_TASK_COUNT` (default 50) games share one network; only one is shown.

Tunable constants are at the top of `tetris-ai.py`.

## Project layout

```
tetris-ai.py       game, GUI and learning code
icon.ico           window icon
weight/            saved networks (*.wgt, git-ignored)
requirements.txt
```

## License

[MIT](LICENSE)

*Tetris is a trademark of The Tetris Company. This is an independent hobby project, not affiliated with or endorsed by it.*
