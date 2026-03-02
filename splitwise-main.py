"""
Splitwise: Efficient Partitioning of LLMs for Edge-Cloud Collaborative Inference
via Lyapunov-assisted Reinforcement Learning

This module implements the core Splitwise framework for dynamic LLM partitioning
between edge devices and cloud servers using Lyapunov optimization theory.

Paper: "Splitwise: Efficient Partitioning of LLMs for Edge-Cloud Collaborative 
       Inference via Lyaponov-assisted RL" (UCC 2025)


GitHub: https://github.com/Abolfazl-Younesi/Splitwise
Paper Link: https://dl.acm.org/doi/10.1145/3773274.3774267
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """System state representation for the MDP formulation"""
    queue_length: float
    queue_avg: float  # Moving average
    bandwidth: float
    bandwidth_avg: float
    bandwidth_var: float
    arrival_rate: float
    arrival_avg: float
    edge_memory_available: float
    edge_compute_available: float
    timestamp: float
    history_embedding: np.ndarray


@dataclass
class PartitionAction:
    """Partition action representation"""
    layer_partitions: List[Dict[str, any]]  # Per-layer partition decisions
    
    def __post_init__(self):
        """Validate partition action"""
        if not self.layer_partitions:
            raise ValueError("Layer partitions cannot be empty")


@dataclass
class PerformanceMetrics:
    """Performance metrics for evaluation"""
    latency: float
    energy: float
    accuracy_loss: float
    communication_overhead: float
    queue_backlog: float


class LyapunovDriftCalculator:
    """Calculates Lyapunov drift for queue stability"""
    
    def __init__(self, queue_weight: float = 1.0):
        self.queue_weight = queue_weight
        
    def compute_drift(self, current_queue: float, arrival_rate: float, 
                     service_rate: float) -> float:
        """
        Compute one-step Lyapunov drift
        
        Args:
            current_queue: Current queue backlog
            arrival_rate: Request arrival rate
            service_rate: Service rate under current partition
            
        Returns:
            Lyapunov drift value
        """
        # Lyapunov function: L(Q) = 0.5 * Q^2
        # Drift: E[L(Q(t+1)) - L(Q(t)) | Q(t)]
        
        # Expected next queue state
        next_queue_expected = max(0, current_queue + arrival_rate - service_rate)
        
        # Drift computation
        current_lyapunov = 0.5 * (current_queue ** 2)
        next_lyapunov = 0.5 * (next_queue_expected ** 2)
        
        drift = next_lyapunov - current_lyapunov
        return drift * self.queue_weight


class CostPredictor(nn.Module):
    """Neural network predictor for latency and energy costs"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2)  # [latency, energy]
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict latency and energy costs"""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class PolicyNetwork(nn.Module):
    """Hierarchical policy network for partition decisions"""
    
    def __init__(self, state_dim: int, num_layers: int, num_heads: int, 
                 hidden_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # LSTM for sequential layer decisions
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Per-layer decision networks
        self.head_predictor = nn.Linear(hidden_dim, num_heads)  # Head placement
        self.ffn_predictor = nn.Linear(hidden_dim, 3)  # FFN placement modes
        
        # Value network for advantage computation
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, state: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate partition decisions
        
        Args:
            state: System state tensor
            temperature: Gumbel-softmax temperature
            
        Returns:
            Tuple of (actions, log_probs)
        """
        batch_size = state.size(0)
        
        # Encode state
        encoded_state = self.state_encoder(state)  # [batch, hidden_dim]
        
        # Expand for all layers
        layer_inputs = encoded_state.unsqueeze(1).repeat(1, self.num_layers, 1)  # [batch, layers, hidden]
        
        # LSTM processing for layer dependencies
        lstm_out, _ = self.lstm(layer_inputs)  # [batch, layers, hidden]
        
        # Generate decisions for each layer
        head_logits = self.head_predictor(lstm_out)  # [batch, layers, heads]
        ffn_logits = self.ffn_predictor(lstm_out)   # [batch, layers, 3]
        
        # Apply Gumbel-softmax for differentiable discrete sampling
        head_probs = torch.sigmoid(head_logits / temperature)  # Binary for each head
        ffn_probs = torch.softmax(ffn_logits / temperature, dim=-1)  # Categorical for FFN
        
        # Sample actions
        head_actions = torch.bernoulli(head_probs)  # [batch, layers, heads]
        ffn_dist = Categorical(ffn_probs)
        ffn_actions = ffn_dist.sample()  # [batch, layers]
        
        # Compute log probabilities
        head_log_probs = (head_actions * torch.log(head_probs + 1e-8) + 
                         (1 - head_actions) * torch.log(1 - head_probs + 1e-8))
        ffn_log_probs = ffn_dist.log_prob(ffn_actions)
        
        # Combine actions
        actions = torch.cat([
            head_actions.reshape(batch_size, -1),
            ffn_actions.unsqueeze(-1)
        ], dim=-1)
        
        # Combine log probabilities
        log_probs = torch.cat([
            head_log_probs.reshape(batch_size, -1).sum(dim=-1, keepdim=True),
            ffn_log_probs.sum(dim=-1, keepdim=True)
        ], dim=-1).sum(dim=-1)
        
        return actions, log_probs
    
    def compute_value(self, state: torch.Tensor) -> torch.Tensor:
        """Compute state value for advantage estimation"""
        encoded_state = self.state_encoder(state)
        return self.value_net(encoded_state).squeeze(-1)


class AdaptiveQuantizer:
    """Adaptive quantization at partition boundaries"""
    
    def __init__(self, sensitivity_threshold: float = 0.1):
        self.sensitivity_threshold = sensitivity_threshold
        self.boundary_stats = {}
        
    def update_sensitivity(self, boundary_id: str, gradient_norm: float, 
                          activation_norm: float):
        """Update sensitivity statistics for a boundary"""
        sensitivity = gradient_norm * activation_norm
        
        if boundary_id not in self.boundary_stats:
            self.boundary_stats[boundary_id] = deque(maxlen=100)
            
        self.boundary_stats[boundary_id].append(sensitivity)
    
    def get_quantization_bits(self, boundary_id: str) -> int:
        """Get appropriate quantization bits for boundary"""
        if boundary_id not in self.boundary_stats:
            return 8  # Default
            
        avg_sensitivity = np.mean(self.boundary_stats[boundary_id])
        
        if avg_sensitivity > self.sensitivity_threshold * 2:
            return 16  # High sensitivity
        elif avg_sensitivity > self.sensitivity_threshold:
            return 8   # Medium sensitivity
        else:
            return 4   # Low sensitivity
    
    def quantize_tensor(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Quantize tensor to specified bits"""
        if bits == 32:
            return tensor
            
        # Simple uniform quantization
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / (2**bits - 1)
        
        quantized = torch.round((tensor - min_val) / scale) * scale + min_val
        return quantized


class SplitWiseFramework:
    """Main Splitwise framework implementation"""
    
    def __init__(self, config: Dict):
        """
        Initialize Splitwise framework
        
        Args:
            config: Configuration dictionary containing model and system parameters
        """
        self.config = config
        
        # Model parameters
        self.num_layers = config['model']['num_layers']
        self.num_heads = config['model']['num_heads']
        self.state_dim = config['system']['state_dim']
        
        # Initialize components
        self._initialize_components()
        
        # Training parameters
        self.lr = config['training']['learning_rate']
        self.gamma = config['training']['discount_factor']
        self.v_min = config['training']['v_min']
        self.v_max = config['training']['v_max']
        self.temperature = config['training']['initial_temperature']
        self.temp_decay = config['training']['temperature_decay']
        
        # History tracking
        self.state_history = deque(maxlen=config['system']['history_length'])
        self.performance_history = deque(maxlen=1000)
        
        logger.info("Splitwise framework initialized successfully")
    
    def _initialize_components(self):
        """Initialize all framework components"""
        # Neural networks
        self.policy_net = PolicyNetwork(
            state_dim=self.state_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.config['model']['hidden_dim']
        )
        
        action_dim = self.num_layers * (self.num_heads + 1)  # Heads + FFN per layer
        self.cost_predictor = CostPredictor(
            state_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=self.config['model']['hidden_dim']
        )
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.cost_optimizer = optim.Adam(self.cost_predictor.parameters(), lr=self.lr * 0.1)
        
        # Other components
        self.lyapunov_calculator = LyapunovDriftCalculator()
        self.quantizer = AdaptiveQuantizer()
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.stability_violations = []
    
    def encode_state(self, raw_state: Dict) -> SystemState:
        """Encode raw system state into structured format"""
        # Update moving averages
        if len(self.state_history) > 0:
            prev_state = self.state_history[-1]
            queue_avg = 0.9 * prev_state.queue_avg + 0.1 * raw_state['queue_length']
            bandwidth_avg = 0.9 * prev_state.bandwidth_avg + 0.1 * raw_state['bandwidth']
            arrival_avg = 0.9 * prev_state.arrival_avg + 0.1 * raw_state['arrival_rate']
            
            # Compute bandwidth variance
            bandwidth_var = 0.9 * prev_state.bandwidth_var + 0.1 * (raw_state['bandwidth'] - bandwidth_avg) ** 2
        else:
            queue_avg = raw_state['queue_length']
            bandwidth_avg = raw_state['bandwidth']
            arrival_avg = raw_state['arrival_rate']
            bandwidth_var = 0.0
        
        # Create history embedding (simple average for now)
        if len(self.state_history) > 0:
            recent_states = list(self.state_history)[-10:]  # Last 10 states
            history_features = []
            for state in recent_states:
                history_features.extend([state.queue_length, state.bandwidth, state.arrival_rate])
            history_embedding = np.array(history_features)
            
            # Pad or truncate to fixed size
            target_size = 30  # 10 states * 3 features
            if len(history_embedding) < target_size:
                history_embedding = np.pad(history_embedding, (0, target_size - len(history_embedding)))
            else:
                history_embedding = history_embedding[:target_size]
        else:
            history_embedding = np.zeros(30)
        
        state = SystemState(
            queue_length=raw_state['queue_length'],
            queue_avg=queue_avg,
            bandwidth=raw_state['bandwidth'],
            bandwidth_avg=bandwidth_avg,
            bandwidth_var=bandwidth_var,
            arrival_rate=raw_state['arrival_rate'],
            arrival_avg=arrival_avg,
            edge_memory_available=raw_state['edge_memory_available'],
            edge_compute_available=raw_state['edge_compute_available'],
            timestamp=time.time(),
            history_embedding=history_embedding
        )
        
        self.state_history.append(state)
        return state
    
    def state_to_tensor(self, state: SystemState) -> torch.Tensor:
        """Convert system state to tensor for neural network input"""
        features = [
            state.queue_length,
            state.queue_avg,
            state.bandwidth,
            state.bandwidth_avg,
            state.bandwidth_var,
            state.arrival_rate,
            state.arrival_avg,
            state.edge_memory_available,
            state.edge_compute_available
        ]
        
        # Add history embedding
        features.extend(state.history_embedding.tolist())
        
        return torch.FloatTensor(features).unsqueeze(0)
    
    def decode_action(self, action_tensor: torch.Tensor) -> PartitionAction:
        """Decode neural network output to partition action"""
        action = action_tensor.squeeze(0).detach().numpy()
        
        layer_partitions = []
        idx = 0
        
        for layer in range(self.num_layers):
            # Extract head assignments
            head_assignments = action[idx:idx + self.num_heads]
            head_assignments = (head_assignments > 0.5).astype(int)
            
            # Extract FFN assignment
            ffn_assignment = int(action[idx + self.num_heads])
            
            layer_partition = {
                'layer_id': layer,
                'head_assignments': head_assignments.tolist(),  # 0=edge, 1=cloud
                'ffn_assignment': ffn_assignment,  # 0=edge, 1=cloud, 2=split
                'edge_heads': np.sum(head_assignments == 0),
                'cloud_heads': np.sum(head_assignments == 1)
            }
            
            layer_partitions.append(layer_partition)
            idx += self.num_heads + 1
        
        return PartitionAction(layer_partitions=layer_partitions)
    
    def compute_reward(self, state: SystemState, action: PartitionAction, 
                      metrics: PerformanceMetrics) -> float:
        """
        Compute Lyapunov-guided reward
        
        Args:
            state: Current system state
            action: Taken partition action
            metrics: Observed performance metrics
            
        Returns:
            Reward value
        """
        # Service rate based on latency
        service_rate = 1000.0 / metrics.latency if metrics.latency > 0 else 0
        
        # Compute Lyapunov drift
        drift = self.lyapunov_calculator.compute_drift(
            current_queue=state.queue_length,
            arrival_rate=state.arrival_rate,
            service_rate=service_rate
        )
        
        # Immediate cost
        cost_weights = self.config['reward']['cost_weights']
        immediate_cost = (
            cost_weights['latency'] * metrics.latency +
            cost_weights['energy'] * metrics.energy +
            cost_weights['accuracy'] * metrics.accuracy_loss
        )
        
        # Adaptive V parameter based on queue length
        v_current = self.v_min + (self.v_max - self.v_min) * np.exp(-state.queue_length / 50.0)
        
        # Lyapunov-guided reward
        reward = -(v_current * drift + immediate_cost)
        
        return reward
    
    def select_action(self, state: SystemState, training: bool = True) -> Tuple[PartitionAction, torch.Tensor]:
        """
        Select partition action using current policy
        
        Args:
            state: Current system state
            training: Whether in training mode
            
        Returns:
            Tuple of (partition_action, log_prob)
        """
        state_tensor = self.state_to_tensor(state)
        
        with torch.no_grad() if not training else torch.enable_grad():
            action_tensor, log_prob = self.policy_net(state_tensor, self.temperature)
        
        action = self.decode_action(action_tensor)
        
        return action, log_prob
    
    def update_policy(self, states: List[SystemState], actions: List[PartitionAction],
                     rewards: List[float], log_probs: List[torch.Tensor]):
        """
        Update policy using PPO-style optimization
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            log_probs: Batch of action log probabilities
        """
        # Convert to tensors
        state_tensors = torch.stack([self.state_to_tensor(s) for s in states]).squeeze(1)
        rewards_tensor = torch.FloatTensor(rewards)
        old_log_probs = torch.stack(log_probs)
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards_tensor)
        values = self.policy_net.compute_value(state_tensors)
        advantages = returns - values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.config['training']['ppo_epochs']):
            # Forward pass
            _, new_log_probs = self.policy_net(state_tensors, self.temperature)
            new_values = self.policy_net.compute_value(state_tensors)
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            
            # Clipped surrogate objective
            clip_param = self.config['training']['clip_param']
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = ((new_values - returns) ** 2).mean()
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
            
            # Update
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
        
        # Update temperature
        self.temperature = max(0.1, self.temperature * self.temp_decay)
        
        logger.info(f"Policy updated. Loss: {total_loss.item():.4f}, Temperature: {self.temperature:.4f}")
    
    def _compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'cost_predictor': self.cost_predictor.state_dict(),
            'config': self.config,
            'temperature': self.temperature
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.cost_predictor.load_state_dict(checkpoint['cost_predictor'])
        self.temperature = checkpoint.get('temperature', 1.0)
        logger.info(f"Model loaded from {filepath}")


def create_default_config() -> Dict:
    """Create default configuration for Splitwise"""
    return {
        'model': {
            'num_layers': 32,
            'num_heads': 32,
            'hidden_dim': 512
        },
        'system': {
            'state_dim': 39,  # 9 basic features + 30 history features
            'history_length': 100
        },
        'training': {
            'learning_rate': 3e-4,
            'discount_factor': 0.99,
            'v_min': 0.1,
            'v_max': 10.0,
            'initial_temperature': 1.0,
            'temperature_decay': 0.995,
            'ppo_epochs': 4,
            'clip_param': 0.2
        },
        'reward': {
            'cost_weights': {
                'latency': 1.0,
                'energy': 0.5,
                'accuracy': 2.0
            }
        }
    }

