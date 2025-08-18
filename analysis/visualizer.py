import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Load logs
logs = []
with open("nn_decision_logs.json", "r") as file:
    for line in file:
        logs.append(json.loads(line))

# Group actions and rewards by player total and softness
actions_by_total_soft = defaultdict(lambda: defaultdict(list))
rewards_by_action_total_soft = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for log in logs:
    player_total = log["state"][0]  # Assuming the player's total is the first element in the state
    player_is_soft = log["state"][1]  # Assuming "softness" is the second element
    dealer_card_value = log["state"][2]
    has_doubled = log["state"][3]

    chosen_action = log["chosen_action"]
    reward = log["reward"]

    # Group actions and rewards by total and softness
    actions_by_total_soft[player_total][player_is_soft].append(chosen_action)
    rewards_by_action_total_soft[player_total][player_is_soft][chosen_action].append(reward)

# Specify player total and softness
player_total = 12
player_is_soft = 1 # Set to 1 for soft hands, 0 for hard hands

if player_total in actions_by_total_soft and player_is_soft in actions_by_total_soft[player_total]:
    # Get action counts for the specified condition
    action_counts = Counter(actions_by_total_soft[player_total][player_is_soft])

    # Calculate average rewards for each action
    average_rewards = {
        action: sum(rewards) / len(rewards) if rewards else 0
        for action, rewards in rewards_by_action_total_soft[player_total][player_is_soft].items()
    }

    # Plot action counts
    plt.bar(action_counts.keys(), action_counts.values(), tick_label=["Hit", "Stand", "Double"])

    # Annotate with average rewards
    for i, (action, count) in enumerate(action_counts.items()):
        avg_reward = average_rewards[action]
        plt.text(i, count, f"Avg R: {avg_reward:.2f}", ha="center", va="bottom")

    # Customize graph
    softness_label = "Soft" if player_is_soft else "Hard"
    plt.title(f"Action Distribution and Average Rewards for {softness_label} Player Total = {player_total}")
    plt.xlabel("Actions")
    plt.ylabel("Count")
    plt.show()
else:
    softness_label = "Soft" if player_is_soft else "Hard"
    print(f"No data for {softness_label} player total = {player_total}")
