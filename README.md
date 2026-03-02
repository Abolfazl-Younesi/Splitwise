# Splitwise

**Efficient Partitioning of LLMs for Edge-Cloud Collaborative Inference via Lyapunov-assisted Reinforcement Learning**

This repository contains an implementation of the **Splitwise** framework algorithm only described in the UCC 2025 paper: "Splitwise: Efficient Partitioning of LLMs for Edge-Cloud Collaborative Inference via Lyaponov-assisted RL". The core idea is to dynamically partition large language models (LLMs) between resource-constrained edge devices and powerful cloud servers by leveraging Lyapunov optimization theory within a reinforcement learning loop.

## 📌 Features

- **Dynamic LLM partitioning** between edge and cloud based on system state.
- **Lyapunov-guided reward** structure ensuring queue stability and low latency.
- **Hierarchical policy network** with per-layer decisions (attention heads and FFN) using Gumbel-softmax for differentiable discrete actions.
- **Adaptive quantization** at partition boundaries to reduce communication overhead without sacrificing accuracy.
- **Cost predictor network** estimating latency and energy for chosen partitions.

## 🧠 Core Components

| Module | Purpose |
|--------|---------|
| `SystemState` | Encapsulates the state of the system for MDP input. |
| `PartitionAction` | Represents layer-wise partition decisions. |
| `LyapunovDriftCalculator` | Computes Lyapunov drift for stability. |
| `CostPredictor` | Neural network predicting latency/energy costs. |
| `PolicyNetwork` | Hierarchical policy generating partition actions. |
| `AdaptiveQuantizer` | Adjusts quantization based on sensitivity stats. |
| `SplitWiseFramework` | Main class tying components together and managing training.

## 🛠️ Usage

1. **Install Dependencies**

   ```sh
   pip install -r requirements.txt
   ```

2. **Configuration**

   see JSON file containing the following sections:

   - `model`: `num_layers`, `num_heads`, `hidden_dim`.
   - `system`: `state_dim`, `history_length`.
   - `training`: `learning_rate`, `discount_factor`, `v_min`, `v_max`, `initial_temperature`, `temperature_decay`.
   - `reward`: `cost_weights` for latency, energy, and accuracy.


3. **Initialize Framework**

   ```python
   from splitwise_main import SplitWiseFramework
   config = ...  # load dictionary
   framework = SplitWiseFramework(config)
   ```

4. **Training Loop**

   - Generate or simulate `SystemState` objects.
   - Call `framework.select_action(state)` to obtain partition decisions.
   - Evaluate the chosen partition to produce `PerformanceMetrics` (latency, energy, accuracy loss, etc.).
   - Compute rewards with `framework.compute_reward(...)`.
   - Periodically update policy with `framework.update_policy(...)` and update cost predictor.

## 🚀 Getting Involved

- Clone the GitHub repo: `https://github.com/Abolfazl-Younesi/Splitwise`
- Experiment with different reward weights, model sizes, and quantization schemes.
- Compare edge-cloud partition strategies on real or synthetic workloads.

## 📚 References

- **Paper**: [Splitwise: Efficient Partitioning of LLMs for Edge-Cloud Collaborative Inference via Lyaponov-assisted RL](https://dl.acm.org/doi/10.1145/3773274.3774267)

### 📖 Citation

If you use this work in your research, please cite:

```
@inproceedings{10.1145/3773274.3774267,
  author = {Younesi, Abolfazl and Shabrang Maryan, Abbas and Oustad, Elyas and Najafabadi Samani, Zahra and Ansari, Mohsen and Fahringer, Thomas},
  title = {Splitwise: Collaborative Edge–Cloud Inference for LLMs via Lyapunov-Assisted DRL},
  year = {2026},
  isbn = {9798400722851},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3773274.3774267},
  doi = {10.1145/3773274.3774267},
  abstract = {Deploying large language models (LLMs) on edge devices is challenging due to their limited memory and power resources. Cloud-only inference reduces device burden but introduces high latency and cost. Static edge–cloud partitions optimize a single metric and struggle when bandwidth fluctuates. We propose Splitwise, a novel Lyapunov-assisted deep reinforcement learning (DRL) framework for fine-grained, adaptive partitioning of LLMs across edge and cloud environments. Splitwise decomposes transformer layers into attention heads and feed-forward sub-blocks, exposing exponentially more partition choices than layer-wise schemes. A hierarchical DRL policy, guided by Lyapunov optimization, jointly minimizes latency, energy consumption, and accuracy degradation while guaranteeing queue stability under stochastic workloads and variable network bandwidth. Splitwise also guarantees robustness via partition checkpoints with exponential backoff recovery in case of communication failures. Experiments on Jetson Orin NX, Galaxy S23, and Raspberry Pi 5 with GPT‑2 (1.5B), LLaMA‑7B, and LLaMA‑13B show that Splitwise reduces end‑to‑end latency by 1.4 \texttimes{} –2.8 \texttimes{} and cuts energy consumption by up to 41\% compared with existing partitioners. It lowers the 95th-percentile latency by 53–61\% relative to cloud-only execution, while maintaining accuracy and modest memory requirements.},
  booktitle = {Proceedings of the 18th IEEE/ACM International Conference on Utility and Cloud Computing},
  articleno = {12},
  numpages = {11},
  keywords = {Large Language Model, Edge-Cloud Inference, Lyapunov Optimization, Deep Reinforcement Learning},
  location = {
  },
  series = {UCC '25}
}
```


---

the code in this workspace provides a baseline for research and experimentation. Use it as a starting point for extending partition strategies, exploring reinforcement learning techniques, or integrating with real edge-cloud systems.

