import re
from typing import Any, Dict, Tuple

from app.schemas.paper import Paper
from app.services.identifier_utils import extract_arxiv_id_from_url


LOCAL_LANDMARKS = {
    "transformers": [
        {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "published_date": "2017-06-12",
            "url": "https://arxiv.org/abs/1706.03762",
            "topic_tags": ["Transformers", "NLP"],
            "subtopic": "sequence modeling",
            "seminal_score": 1.0,
            "survey_flag": False,
            "venue": "NeurIPS",
            "aliases": ["attention is all you need", "transformer architecture"],
        },
        {"title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "authors": ["Jacob Devlin"], "published_date": "2018-10-11", "url": "https://arxiv.org/abs/1810.04805", "topic_tags": ["Transformers", "BERT"]},
        {"title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach", "authors": ["Yinhan Liu"], "published_date": "2019-07-26", "url": "https://arxiv.org/abs/1907.11692", "topic_tags": ["Transformers", "RoBERTa"]},
        {"title": "Language Models are Few-Shot Learners", "authors": ["Tom Brown"], "published_date": "2020-05-28", "url": "https://arxiv.org/abs/2005.14165", "topic_tags": ["Transformers", "GPT"]},
        {"title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", "authors": ["Colin Raffel"], "published_date": "2019-10-23", "url": "https://arxiv.org/abs/1910.10683", "topic_tags": ["Transformers", "T5"]},
    ],
    "diffusion models": [
        {
            "title": "Denoising Diffusion Probabilistic Models",
            "authors": ["Jonathan Ho", "Ajay Jain", "Pieter Abbeel"],
            "published_date": "2020-06-19",
            "url": "https://arxiv.org/abs/2006.11239",
            "topic_tags": ["Diffusion Models", "Generative Modeling"],
        },
        {"title": "Score-Based Generative Modeling through Stochastic Differential Equations", "authors": ["Yang Song"], "published_date": "2020-06-19", "url": "https://arxiv.org/abs/2011.13456", "topic_tags": ["Diffusion Models"]},
        {"title": "High-Resolution Image Synthesis with Latent Diffusion Models", "authors": ["Robin Rombach"], "published_date": "2021-12-20", "url": "https://arxiv.org/abs/2112.10752", "topic_tags": ["Diffusion Models", "Stable Diffusion"]},
    ],
    "retrieval augmented generation": [
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "authors": ["Patrick Lewis", "Ethan Perez"],
            "published_date": "2020-05-22",
            "url": "https://arxiv.org/abs/2005.11401",
            "topic_tags": ["RAG", "NLP", "Information Retrieval"],
        },
        {"title": "Dense Passage Retrieval for Open-Domain Question Answering", "authors": ["Vladimir Karpukhin"], "published_date": "2020-04-26", "url": "https://arxiv.org/abs/2004.04906", "topic_tags": ["RAG", "Dense Retrieval"]},
        {"title": "REALM: Retrieval-Augmented Language Model Pre-Training", "authors": ["Kelvin Guu"], "published_date": "2020-02-14", "url": "https://arxiv.org/abs/2002.08909", "topic_tags": ["RAG", "REALM"]},
        {"title": "Fusion-in-Decoder for Open-Domain Question Answering", "authors": ["Gautier Izacard"], "published_date": "2020-07-15", "url": "https://arxiv.org/abs/2007.01282", "topic_tags": ["RAG", "FiD"]},
    ],
    "reinforcement learning": [
        {
            "title": "Playing Atari with Deep Reinforcement Learning",
            "authors": ["Volodymyr Mnih", "Koray Kavukcuoglu"],
            "published_date": "2013-12-19",
            "url": "https://arxiv.org/abs/1312.5602",
            "topic_tags": ["Reinforcement Learning", "DQN"],
        },
        {"title": "Human-level Control through Deep Reinforcement Learning", "authors": ["Volodymyr Mnih"], "published_date": "2015-02-25", "url": "https://www.nature.com/articles/nature14236", "topic_tags": ["Reinforcement Learning", "DQN"]},
        {"title": "Proximal Policy Optimization Algorithms", "authors": ["John Schulman"], "published_date": "2017-07-20", "url": "https://arxiv.org/abs/1707.06347", "topic_tags": ["Reinforcement Learning", "PPO"]},
        {"title": "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning", "authors": ["Tuomas Haarnoja"], "published_date": "2018-01-05", "url": "https://arxiv.org/abs/1801.01290", "topic_tags": ["Reinforcement Learning", "SAC"]},
    ],
    "graph neural networks": [
        {
            "title": "Semi-Supervised Classification with Graph Convolutional Networks",
            "authors": ["Thomas Kipf", "Max Welling"],
            "published_date": "2016-09-09",
            "url": "https://arxiv.org/abs/1609.02907",
            "topic_tags": ["Graph Neural Networks", "GCN"],
        },
        {"title": "Graph Attention Networks", "authors": ["Petar Velickovic"], "published_date": "2017-10-30", "url": "https://arxiv.org/abs/1710.10903", "topic_tags": ["Graph Neural Networks", "GAT"]},
        {"title": "Inductive Representation Learning on Large Graphs", "authors": ["Will Hamilton"], "published_date": "2017-09-07", "url": "https://arxiv.org/abs/1706.02216", "topic_tags": ["Graph Neural Networks", "GraphSAGE"]},
        {"title": "How Powerful are Graph Neural Networks?", "authors": ["Keyulu Xu"], "published_date": "2018-12-26", "url": "https://arxiv.org/abs/1810.00826", "topic_tags": ["Graph Neural Networks"]},
    ],
    "large language models": [
        {
            "title": "Language Models are Few-Shot Learners",
            "authors": ["Tom Brown", "Benjamin Mann"],
            "published_date": "2020-05-28",
            "url": "https://arxiv.org/abs/2005.14165",
            "topic_tags": ["Large Language Models", "GPT"],
        },
        {"title": "PaLM: Scaling Language Modeling with Pathways", "authors": ["Aakanksha Chowdhery"], "published_date": "2022-04-04", "url": "https://arxiv.org/abs/2204.02311", "topic_tags": ["Large Language Models", "PaLM"]},
        {"title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "authors": ["Jason Wei"], "published_date": "2022-01-28", "url": "https://arxiv.org/abs/2201.11903", "topic_tags": ["Large Language Models", "Prompting"]},
    ],
    "generative adversarial networks": [
        {
            "title": "Generative Adversarial Nets",
            "authors": ["Ian Goodfellow", "Jean Pouget-Abadie"],
            "published_date": "2014-06-10",
            "url": "https://arxiv.org/abs/1406.2661",
            "topic_tags": ["GANs", "Generative Modeling"],
        },
        {"title": "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", "authors": ["Jun-Yan Zhu"], "published_date": "2017-03-30", "url": "https://arxiv.org/abs/1703.10593", "topic_tags": ["GANs", "CycleGAN"]},
        {"title": "Image-to-Image Translation with Conditional Adversarial Networks", "authors": ["Phillip Isola"], "published_date": "2016-11-07", "url": "https://arxiv.org/abs/1611.07004", "topic_tags": ["GANs", "pix2pix"]},
        {"title": "Progressive Growing of GANs for Improved Quality, Stability, and Variation", "authors": ["Tero Karras"], "published_date": "2017-10-26", "url": "https://arxiv.org/abs/1710.10196", "topic_tags": ["GANs", "StyleGAN"]},
    ],
    "contrastive learning": [
        {
            "title": "A Simple Framework for Contrastive Learning of Visual Representations",
            "authors": ["Ting Chen", "Simon Kornblith"],
            "published_date": "2020-02-13",
            "url": "https://arxiv.org/abs/2002.05709",
            "topic_tags": ["Contrastive Learning", "SimCLR"],
        },
        {"title": "Momentum Contrast for Unsupervised Visual Representation Learning", "authors": ["Kaiming He"], "published_date": "2019-11-22", "url": "https://arxiv.org/abs/1911.05722", "topic_tags": ["Contrastive Learning", "MoCo"]},
        {"title": "Learning Transferable Visual Models From Natural Language Supervision", "authors": ["Alec Radford"], "published_date": "2021-02-26", "url": "https://arxiv.org/abs/2103.00020", "topic_tags": ["Contrastive Learning", "CLIP"]},
        {"title": "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning", "authors": ["Jean-Bastien Grill"], "published_date": "2020-06-14", "url": "https://arxiv.org/abs/2006.07733", "topic_tags": ["Contrastive Learning", "BYOL"]},
    ],
}


def _norm_registry_title(title: str) -> str:
    # Match ``ranking_service._norm_title`` so registry keys align with ranking.
    return re.sub(r"[^a-z0-9 ]", "", (title or "").lower()).strip()


def _build_landmark_registry_index() -> Dict[str, Dict[str, Any]]:
    """Flatten ``LOCAL_LANDMARKS`` into a normalized-title → metadata map."""
    idx: Dict[str, Dict[str, Any]] = {}
    for topic, rows in LOCAL_LANDMARKS.items():
        for row in rows:
            meta = {
                "topic": topic,
                "title": row["title"],
                "subtopic": row.get("subtopic", ""),
                "seminal_score": float(row.get("seminal_score", 1.0)),
                "survey_flag": bool(row.get("survey_flag", False)),
                "venue": row.get("venue", ""),
                "aliases": list(row.get("aliases", [])),
            }
            nt = _norm_registry_title(row["title"])
            if nt:
                idx[nt] = meta
            for a in meta["aliases"]:
                an = _norm_registry_title(a)
                if an:
                    idx[an] = meta
    return idx


LANDMARK_REGISTRY_INDEX: Dict[str, Dict[str, Any]] = _build_landmark_registry_index()


def registry_boost_and_survey_flag(norm_title: str) -> Tuple[float, bool]:
    """
    Return (ranking boost from curated registry, survey_flag) for a normalized title.

    ``norm_title`` must match :func:`ranking_service._norm_title` / registry keys.
    """
    meta = LANDMARK_REGISTRY_INDEX.get(norm_title)
    if not meta:
        return 0.0, False
    boost = 0.12 * max(0.0, min(1.5, float(meta.get("seminal_score", 1.0))))
    return boost, bool(meta.get("survey_flag"))


def fallback_landmarks_for_topic(topic: str) -> list[Paper]:
    key = (topic or "").lower().strip()
    rows = LOCAL_LANDMARKS.get(key, [])
    if not rows:
        for k, v in LOCAL_LANDMARKS.items():
            if key in k or k in key:
                rows = v
                break
    papers: list[Paper] = []
    for i, row in enumerate(rows):
        url = row.get("url", "")
        aid = extract_arxiv_id_from_url(url)
        papers.append(
            Paper(
                id=f"local-{key.replace(' ', '-')}-{i+1}",
                title=row["title"],
                abstract=f"Locally curated landmark paper for {key}.",
                authors=row.get("authors", []),
                published_date=row.get("published_date", ""),
                source="local_landmark",
                url=url,
                doi="",
                arxiv_id=aid,
                openalex_work_id="",
                citations=int(2000 * float(row.get("seminal_score", 1.0)) + 1000),
                topic_tags=list(row.get("topic_tags", [])),
                venue=row.get("venue") or "Local Landmark",
            )
        )
    return papers
