"""
Retrieval-Augmented Deepfake Detection (RAG).
FAISS-based artifact database for storing and retrieving known deepfake patterns.
Retrieved artifact context is fused into the classifier via the multimodal transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RAGRetrieval(nn.Module):
    """
    Retrieval-Augmented Detection module.

    1. Projects model embeddings to 512d query space
    2. Queries FAISS index for similar artifact patterns
    3. Aggregates retrieved embeddings via attention-weighted mean
    4. Produces a 256d context vector for the fusion transformer
    """

    def __init__(
        self,
        input_dim: int = 1280,
        query_dim: int = 512,
        output_dim: int = 256,
        top_k: int = 8,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.output_dim = output_dim
        self.top_k = top_k

        # Projection to query embedding space
        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, query_dim),
            nn.ReLU(),
            nn.Linear(query_dim, query_dim),
        )

        # Attention aggregation over retrieved results
        self.attn_query = nn.Linear(query_dim, query_dim)
        self.attn_key = nn.Linear(query_dim, query_dim)
        self.attn_value = nn.Linear(query_dim, query_dim)
        self.attn_scale = query_dim ** -0.5

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(query_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # The FAISS index will be set externally (not a nn parameter)
        self.faiss_index = None
        self.stored_embeddings = None  # numpy array of stored embeddings
        self.metadata = []  # list of dicts with metadata per embedding

        logger.info(f"RAGRetrieval: query_dim={query_dim}, top_k={top_k}")

    def build_index(self, embeddings: np.ndarray, metadata: list = None):
        """
        Build FAISS index from a set of embeddings.

        Args:
            embeddings: (N, query_dim) normalized embedding vectors
            metadata: list of N dicts with metadata per embedding
        """
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed, RAG will return zeros. Install: pip install faiss-cpu")
            return

        N, D = embeddings.shape
        assert D == self.query_dim, f"Embedding dim {D} != query_dim {self.query_dim}"

        # L2 normalize
        faiss.normalize_L2(embeddings)

        # Build inner-product index (cosine similarity on normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(D)
        self.faiss_index.add(embeddings)
        self.stored_embeddings = embeddings
        self.metadata = metadata or [{}] * N

        logger.info(f"FAISS index built: {N} vectors, dim={D}")

    def save_index(self, path: str):
        """Save FAISS index and metadata to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.faiss_index is not None:
            import faiss
            faiss.write_index(self.faiss_index, str(save_dir / "artifact_db.faiss"))
            np.save(save_dir / "embeddings.npy", self.stored_embeddings)
            with open(save_dir / "metadata.json", "w") as f:
                json.dump(self.metadata, f)
            logger.info(f"RAG index saved to {save_dir}")

    def load_index(self, path: str):
        """Load FAISS index and metadata from disk."""
        load_dir = Path(path)
        try:
            import faiss
            self.faiss_index = faiss.read_index(str(load_dir / "artifact_db.faiss"))
            self.stored_embeddings = np.load(load_dir / "embeddings.npy")
            with open(load_dir / "metadata.json", "r") as f:
                self.metadata = json.load(f)
            logger.info(f"RAG index loaded: {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Could not load RAG index: {e}")

    @torch.no_grad()
    def retrieve(self, query_embedding: torch.Tensor) -> dict:
        """
        Retrieve top-k similar artifacts from FAISS.

        Args:
            query_embedding: (B, query_dim) normalized query vectors

        Returns:
            dict with retrieved embeddings and metadata
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            B = query_embedding.size(0)
            return {
                "retrieved_embeddings": torch.zeros(B, self.top_k, self.query_dim, device=query_embedding.device),
                "distances": torch.zeros(B, self.top_k, device=query_embedding.device),
                "metadata": [[{}] * self.top_k] * B,
            }

        query_np = query_embedding.cpu().numpy().astype(np.float32)

        import faiss
        faiss.normalize_L2(query_np)

        distances, indices = self.faiss_index.search(query_np, self.top_k)

        # Gather retrieved embeddings
        retrieved = self.stored_embeddings[indices.flatten()].reshape(
            query_np.shape[0], self.top_k, self.query_dim
        )
        retrieved_tensor = torch.from_numpy(retrieved).to(query_embedding.device)

        # Gather metadata
        retrieved_meta = []
        for batch_indices in indices:
            batch_meta = [self.metadata[int(i)] if int(i) < len(self.metadata) else {} for i in batch_indices]
            retrieved_meta.append(batch_meta)

        return {
            "retrieved_embeddings": retrieved_tensor,
            "distances": torch.from_numpy(distances).to(query_embedding.device),
            "metadata": retrieved_meta,
        }

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Full RAG forward: project → retrieve → aggregate → output.

        Args:
            spatial_features: (B, input_dim) from spatial encoder

        Returns:
            rag_context: (B, output_dim) retrieved context vector
        """
        # Project to query space
        query = self.query_proj(spatial_features)
        query = F.normalize(query, dim=-1)

        # Retrieve from FAISS
        retrieved = self.retrieve(query)
        retrieved_emb = retrieved["retrieved_embeddings"]  # (B, top_k, query_dim)

        # Attention-weighted aggregation
        q = self.attn_query(query).unsqueeze(1)  # (B, 1, query_dim)
        k = self.attn_key(retrieved_emb)           # (B, top_k, query_dim)
        v = self.attn_value(retrieved_emb)           # (B, top_k, query_dim)

        attn_weights = (q @ k.transpose(-2, -1)) * self.attn_scale  # (B, 1, top_k)
        attn_weights = attn_weights.softmax(dim=-1)

        aggregated = (attn_weights @ v).squeeze(1)  # (B, query_dim)

        # Output projection
        rag_context = self.output_proj(aggregated)  # (B, output_dim)
        return rag_context

    def extract_and_store_embedding(self, spatial_features: torch.Tensor, metadata_batch: list):
        """
        Extract embeddings from a batch and add them to the index.
        Used during database population after training.

        Args:
            spatial_features: (B, input_dim) from spatial encoder
            metadata_batch: list of B metadata dicts
        """
        with torch.no_grad():
            query = self.query_proj(spatial_features)
            query = F.normalize(query, dim=-1)
            embeddings_np = query.cpu().numpy().astype(np.float32)

        if self.stored_embeddings is None:
            self.stored_embeddings = embeddings_np
            self.metadata = metadata_batch
        else:
            self.stored_embeddings = np.concatenate([self.stored_embeddings, embeddings_np], axis=0)
            self.metadata.extend(metadata_batch)
