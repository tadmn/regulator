Setup
```
brew install libsndfile
uv sync
source .venv/bin/activate
```

Train Model
```
cmake modules -B modules/build
cmake --build modules/build
python train_model.py
```