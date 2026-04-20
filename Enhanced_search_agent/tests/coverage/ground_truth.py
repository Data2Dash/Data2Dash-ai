"""
Ground Truth Dataset for Coverage Testing
==========================================
Each entry defines a *topic* the user might search for and a curated list of
landmark papers that a good hybrid search agent SHOULD retrieve.

The `must_find`  list = papers so canonical that any decent retrieval system
                         should return them (high-priority recall).

The `should_find` list = important but slightly less universal papers
                          (good-to-have recall).

Paper matching is done by normalised-title substring match so minor wording
differences (capitalisation, punctuation) are tolerated.
"""

GROUND_TRUTH = [
    {
        "topic": "transformers",
        "description": "Neural network architecture based on self-attention",
        "must_find": [
            "attention is all you need",           # Vaswani et al. 2017
        ],
        "should_find": [
            "bert",                                 # Devlin et al. 2018
            "gpt",                                  # Radford et al.
            "roberta",                              # Liu et al. 2019
            "language models are few-shot learners",# GPT-3
            "t5",                                   # Raffel et al. 2020
        ],
    },
    {
        "topic": "diffusion models",
        "description": "Score-based generative models for image synthesis",
        "must_find": [
            "denoising diffusion probabilistic models",  # Ho et al. 2020
        ],
        "should_find": [
            "score-based generative modeling",      # Song & Ermon 2019
            "high-resolution image synthesis",      # Rombach / LDM
            "dall-e",                               # OpenAI
            "stable diffusion",
            "ddim",                                 # accelerated sampling
        ],
    },
    {
        "topic": "retrieval augmented generation",
        "description": "RAG — combining retrieval with LLM generation",
        "must_find": [
            "retrieval-augmented generation",       # Lewis et al. 2020
        ],
        "should_find": [
            "dense passage retrieval",              # DPR, Karpukhin et al. 2020
            "realm",                                # Guu et al. 2020
            "fusion-in-decoder",
            "rag",
        ],
    },
    {
        "topic": "reinforcement learning",
        "description": "Learning via reward signal and policies",
        "must_find": [
            "playing atari with deep reinforcement learning",  # Mnih et al. 2013
        ],
        "should_find": [
            "human-level control through deep reinforcement learning",  # DQN Nature
            "proximal policy optimization",         # PPO Schulman 2017
            "soft actor-critic",                    # SAC Haarnoja 2018
            "alphago",
        ],
    },
    {
        "topic": "graph neural networks",
        "description": "Deep learning on graph-structured data",
        "must_find": [
            "semi-supervised classification with graph convolutional networks",  # Kipf & Welling 2017
        ],
        "should_find": [
            "graph attention networks",             # Veličković 2018
            "inductive representation learning",    # GraphSAGE
            "how powerful are graph neural networks",
            "graph isomorphism",
        ],
    },
    {
        "topic": "large language models",
        "description": "Foundation models trained on massive text corpora",
        "must_find": [
            "language models are few-shot learners",  # GPT-3 Brown et al. 2020
        ],
        "should_find": [
            "llama",                                # Meta
            "palm",                                 # Google
            "chatgpt",
            "instruction tuning",
            "chain-of-thought prompting",
        ],
    },
    {
        "topic": "generative adversarial networks",
        "description": "GAN-based generative models",
        "must_find": [
            "generative adversarial nets",          # Goodfellow et al. 2014
        ],
        "should_find": [
            "conditional image synthesis",
            "progressive growing of gans",
            "stylegan",
            "pix2pix",
            "cycle-consistent",
        ],
    },
    {
        "topic": "contrastive learning",
        "description": "Self-supervised representation learning via contrastive loss",
        "must_find": [
            "a simple framework for contrastive learning",  # SimCLR Chen et al. 2020
        ],
        "should_find": [
            "momentum contrast",                    # MoCo He et al. 2020
            "clip",                                 # Radford 2021
            "byol",
            "simclr",
        ],
    },
]
