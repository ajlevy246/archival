"""Convert local ArXiv metadata to embeddings."""
from sentence_transformers import SentenceTransformer

"""https://huggingface.co/google/embeddinggemma-2"""
model = SentenceTransformer(
    model_name_or_path="google/embeddinggemma-2", 
    config_kwargs={"vision_config": None, "audio_config": None} # no need for vision/audio encoders
)

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium."
]
embeddings = model.encode(sentences) # valid dimensions: 768, 512, 256, and 128
# model.encode(f"title: {title} | text: {document}")
# query_emb = model.encode(
#    query,
#    truncate_dim=128, # or 512, 256
#    normalize_embeddings=True,
# )

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)