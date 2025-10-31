#!/bin/bash

python - << 'EOF' 
import torch, PIL, gradio, sentence_transformers, pinecone 
print("Torch:", torch.__version__) 
print("PIL:",PIL.__version__)
print("Gradio:",gradio.__version__)
print("SentenceTR:", sentence_transformers.__version__)
EOF
