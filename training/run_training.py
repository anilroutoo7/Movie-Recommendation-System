"""
Download TMDB dataset and train the recommendation model.
Uses Configuration 3 (Fast Training - 10K movies) for quick setup.
"""
import kagglehub
import nltk
import sys
import os
import io

# Fix Windows console encoding for Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Download the TMDB dataset from Kaggle
print("\n[DOWNLOAD] Downloading TMDB Movies Dataset from Kaggle...")
print("(This may take a few minutes on first run)\n")

path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
print(f"Dataset downloaded to: {path}")

# List files in the downloaded directory
for f in os.listdir(path):
    filepath = os.path.join(path, f)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  - {f} ({size_mb:.1f} MB)")

# Import the trainer
from train import MovieRecommenderTrainer

# Fast Training config (10K movies) - takes ~2 minutes, needs ~4GB RAM
print("\n[TRAIN] Starting training with Fast config (10K movies)...")
trainer = MovieRecommenderTrainer(
    output_dir='./models',
    use_dimensionality_reduction=False  # Not needed for small dataset
)

df, sim_matrix = trainer.train(
    path,
    quality_threshold='high',   # 500+ votes (high quality only)
    max_movies=10000             # Top 10K movies
)

print(f"\n[DONE] Final Statistics:")
print(f"   Movies in model: {len(df):,}")
print(f"   Similarity matrix: {sim_matrix.shape}")
print(f"   Memory usage: {sim_matrix.nbytes / 1024**2:.1f} MB")
print(f"\n[SUCCESS] Training complete! You can now run: python manage.py runserver")
