# Connect4 using MCTS with UCT

This project compares various AI strategies in a two-player board game setting using automated head-to-head matchups. The implemented models include both heuristic-based and Monte Carlo Tree Search (MCTS) variants with enhancements like progressive bias, progressive widening, and dynamic exploration.

## 🔧 Requirements

- Python 3.8 or higher
- `pip`

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/jacekkulma/Connect4-with-MCTS-UCT.git
cd Connect4-with-MCTS-UCT
```
### 2. Create and activate a virtual environment
Windows:
```bash
python3 -m venv venv
.\venv\Scripts\activate 
```
Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Install submodules
Submodule can be used to run Connect4 in player vs player mode. To install run:

```bash
git submodule update --init --recursive