#!/bin/bash
#SBATCH --job-name=reading_comprehension_answer_generation
#SBATCH --output=logs/read_answer_%j.out
#SBATCH --error=logs/read_answer_%j.err
#SBATCH --partition=dsba
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=3321908@studbocconi.it

# ——————————————————————————————
# Move into your repo's root directory (adjust path if needed)
cd ~/llm-justification-evaluation

# Make sure logs and output folders exist
mkdir -p logs
mkdir -p data/generated_data

# Load CUDA module
module load nvidia/cuda-12.4.0

# Set up any custom library paths for Ollama
export PATH=$HOME/bin:$PATH
export LD_LIBRARY_PATH=$HOME/lib/ollama/cuda_v12:$LD_LIBRARY_PATH

# Activate conda environment
source /opt/share/modulefiles/sw/miniconda3/etc/profile.d/conda.sh
conda activate nlp_env

# Start Ollama server in background (once)
ollama serve > ollama_server.log 2>&1 &
sleep 10
echo "ollama served"
# Loop through each model in models.txt
while IFS= read -r model; do
    echo "=== Model: $model ==="

    # Pull the model
    ollama pull "$model"

    # Run your generator script
    python scripts/reading_comprehension_answer_generator.py \
        --model "$model" \
        --range 0:1

    # Remove the model to free up space
    ollama rm "$model"

    echo "-----------------------------"
done < models.txt

# Clean up
conda deactivate
