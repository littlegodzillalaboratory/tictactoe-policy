"""Train and use a tiny demonstration policy."""

from tictactoe_policy import GameConfig
from tictactoe_policy.training import train_policy


result = train_policy(GameConfig(3), hidden_size=16, epochs=5, seed=42)
move = result.policy.choose_move([1, -1, 0, 0, 1, 0, -1, 0, 0])
print(move)
