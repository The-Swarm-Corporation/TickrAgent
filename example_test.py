import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from loguru import logger
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import List

# Hyperparameters
embedding_dim: int = 128
nhead: int = 8
num_layers: int = 2
dropout: float = 0.1
gamma: float = 0.99  # discount factor
alpha: float = 0.5  # pain weight factor
learning_rate: float = 1e-4
vocab_size: int = 30522  # GPT-2 vocab size as an example
batch_size: int = 8

# Load dataset from Hugging Face (IMDB reviews)
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = (
    tokenizer.eos_token
)  # Use the end of sequence token as padding

# Loguru logger setup
logger.add("training_log.log", rotation="500 MB")


# Define the Transformer Model
class TransformerPolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super(TransformerPolicy, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> torch.Tensor:
        src_emb: torch.Tensor = self.embedding(
            src
        )  # (seq_len, batch_size, embedding_dim)
        tgt_emb: torch.Tensor = self.embedding(tgt)
        output: torch.Tensor = self.transformer(src_emb, tgt_emb)
        logits: torch.Tensor = self.fc(output)
        return logits


# Define the RL framework
class RLTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        vocab_size: int,
        gamma: float = 0.99,
        alpha: float = 0.5,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.alpha = alpha
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.pains: List[float] = []

    def select_action(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> torch.Tensor:
        logits = self.model(
            src, tgt
        )  # (seq_len, batch_size, vocab_size)
        probs: torch.Tensor = F.softmax(
            logits, dim=-1
        )  # Convert to probabilities
        m: Categorical = Categorical(probs)
        action: torch.Tensor = m.sample()
        self.log_probs.append(
            m.log_prob(action)
        )  # Store log prob for REINFORCE
        return action

    def store_outcome(self, reward: float, pain: float) -> None:
        self.rewards.append(reward)
        self.pains.append(pain)

    def compute_loss(self) -> torch.Tensor:
        R: float = 0
        policy_loss: List[torch.Tensor] = []
        rewards_discounted: List[float] = []

        # Compute discounted reward for each step
        for reward, pain in zip(
            reversed(self.rewards), reversed(self.pains)
        ):
            R = reward - self.alpha * pain + self.gamma * R
            rewards_discounted.insert(0, R)

        rewards_discounted_tensor = torch.tensor(rewards_discounted)
        rewards_discounted_tensor = (
            rewards_discounted_tensor
            - rewards_discounted_tensor.mean()
        ) / (rewards_discounted_tensor.std() + 1e-6)

        # Compute policy gradient loss
        for log_prob, reward in zip(
            self.log_probs, rewards_discounted_tensor
        ):
            policy_loss.append(-log_prob * reward)

        loss: torch.Tensor = torch.cat(policy_loss).sum()
        return loss

    def update_policy(self) -> None:
        loss = self.compute_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.rewards = []
        self.pains = []
        self.log_probs = []
        logger.info(f"Policy updated. Loss: {loss.item()}")


# Reward and Pain calculation (simplified)
def calculate_reward(sequence: torch.Tensor) -> float:
    # Reward for correct sequence generation (e.g., goal achievement)
    return sum(
        1
        for token in sequence
        if token == tokenizer.convert_tokens_to_ids("good")
    )  # Simplified reward


def calculate_pain(sequence: torch.Tensor) -> float:
    # Pain is incurred for incorrect tokens or other penalties
    return sum(
        1
        for token in sequence
        if token == tokenizer.convert_tokens_to_ids("bad")
    )  # Simplified pain metric


# Tokenize the IMDB dataset
# Tokenize the IMDB dataset
def tokenize_function(examples: dict) -> dict:
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]


# Training loop
def train_rl(
    transformer_model: nn.Module,
    vocab_size: int,
    num_episodes: int = 1000,
) -> None:
    optimizer = optim.Adam(
        transformer_model.parameters(), lr=learning_rate
    )
    rl_trainer = RLTrainer(
        transformer_model,
        optimizer,
        vocab_size,
        gamma=gamma,
        alpha=alpha,
    )

    for episode in range(num_episodes):
        batch = train_dataset[
            episode * batch_size : (episode + 1) * batch_size
        ]
        src = torch.tensor(batch["input_ids"]).transpose(
            0, 1
        )  # (seq_len, batch_size)
        tgt = torch.tensor(batch["input_ids"]).transpose(
            0, 1
        )  # For simplicity, tgt = src

        # Select an action (i.e., generate next token sequence)
        action = rl_trainer.select_action(src, tgt)

        # Calculate reward and pain based on the generated sequence
        reward = calculate_reward(action)
        pain = calculate_pain(action)

        # Store the reward and pain for the episode
        rl_trainer.store_outcome(reward, pain)

        # After every 10 episodes, update the policy using the stored log probs and rewards/pains
        if episode % 10 == 0:  # Update policy every 10 episodes
            rl_trainer.update_policy()

        if episode % 100 == 0:
            logger.info(
                f"Episode {episode}: Policy updated with loss."
            )


# Instantiate the model and start training
transformer_model = TransformerPolicy(
    vocab_size, embedding_dim, nhead, num_layers, dropout
)
logger.info("Starting training...")
train_rl(transformer_model, vocab_size)
logger.info("Training complete.")
