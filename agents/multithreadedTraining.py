from bjcards import *
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

PERFORMANCE_DEBUG = False 
INPUT_HANDS = "../data/card_dataset.json"
LOG_OUTPUT = f"../outputs/logs"
AGENT_OUTPUT = f"../outputs/saved_nns"

LAYER_1_SIZE = 32
LAYER_2_SIZE = 32

# Best batch -0.0231  {"batch_size": 64, "gamma": 0.25, "learning_rate": 0.00005,  "epsilon_decay": 0.999995, "epsilon_min": 0.0005},,
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add batch parameteres here:
parameter_sets = [
    {"batch_size": 64, "gamma": 0.25, "learning_rate": 0.0001,  "epsilon_decay": 0.999995, "epsilon_min": 0.0001},
    {"batch_size": 64, "gamma": 0.25, "learning_rate": 0.0001,  "epsilon_decay": 0.999995, "epsilon_min": 0.00005},
    {"batch_size": 64, "gamma": 0.25, "learning_rate": 0.00005,  "epsilon_decay": 0.999995, "epsilon_min": 0.00005},
    {"batch_size": 64, "gamma": 0.25, "learning_rate": 0.00005,  "epsilon_decay": 0.999985, "epsilon_min": 0.000001},
]


class Fixed_shoe:
    def __init__(self, fixed_dealer_sequence=None, fixed_player_sequence=None):
        self.fixed_dealer_sequence = [[Card(v.value if isinstance(v, Card) else v) for v in hand] for hand in (fixed_dealer_sequence or [])]
        self.fixed_player_sequence = [[Card(v.value if isinstance(v, Card) else v) for v in hand]for hand in (fixed_player_sequence or [])]
        self.dealer_index = 0
        self.player_index = 0
        self.length_of_hand = None

    def deal(self, is_dealer=None, hand_over_signal=None, length_of_hand=0):
        card = None
        self.length_of_hand = length_of_hand
        if hand_over_signal:
            self.dealer_index += 1
            self.player_index += 1

            return None

        if is_dealer:
            if self.dealer_index <= len(self.fixed_dealer_sequence):
                current_hand = self.fixed_dealer_sequence[self.dealer_index]
                if current_hand:
                    card = current_hand[length_of_hand]
        if is_dealer is False:
            if self.player_index <= len(self.fixed_player_sequence):
                current_hand = self.fixed_player_sequence[self.player_index]
                if current_hand:
                    card = current_hand[length_of_hand]


        if not isinstance(card, Card):
            raise TypeError(
                f"Invalid card dealt: {card}. Ensure only Card objects are returned."
                f"\nDealer index: {self.dealer_index}, Player index: {self.player_index}"
                f"\nDealer sequence: {self.fixed_dealer_sequence}"
                f"\nPlayer sequence: {self.fixed_player_sequence}"
            )
        return card




class BlackjackEnvAI:
    def __init__(self, dealer_sequence=None, player_sequence=None):
        self.done = False
        self.shoe = Fixed_shoe(fixed_dealer_sequence=dealer_sequence, fixed_player_sequence=player_sequence)
        self.player_hand = Hand() 
        self.dealer_hand = Hand() 
        self.reset()

    def is_hand_over(self):
        if self.done:
            return True
        return False

    def reset(self):
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.player_hand.add_card(self.shoe.deal(is_dealer=False, length_of_hand=len(self.player_hand.cards)))
        self.player_hand.add_card(self.shoe.deal(is_dealer=False, length_of_hand=len(self.player_hand.cards)))
        self.dealer_hand.add_card(self.shoe.deal(is_dealer=True, length_of_hand=len(self.dealer_hand.cards)))
        self.dealer_hand.add_card(self.shoe.deal(is_dealer=True, length_of_hand=len(self.dealer_hand.cards)))
        self.done = False

        return self.get_state_representation()

    def get_state_representation(self):
        player_total = self.player_hand.hand_value()
        is_soft = 1 if self.player_hand.is_soft and player_total <= 21 else 0
        dealer_card_value = self.dealer_hand.cards[0].value
        has_doubled = 1 if self.player_hand.has_doubled else 0

        state_representation = [
            player_total,
            is_soft,
            dealer_card_value,
            has_doubled,
        ]

        return torch.tensor(state_representation, dtype=torch.float32).to(device)

    def check_player_blackjack(self):
        if len(self.player_hand.cards) == 2 and (self.player_hand.hand_value() == 21):
            return True
        return False

    def check_dealer_blackjack(self):
        if len(self.dealer_hand.cards) == 2 and (self.dealer_hand.hand_value() == 21):
            return True

    def get_valid_actions(self):
        # Action 0: Hit, Action 1: Stand, Action 2: Double Down, Action 3: Split
        valid_actions = [0, 1]
        if len(self.player_hand.cards) == 2:
            valid_actions.append(2)
        if self.player_hand.hand_is_splitable():
            valid_actions.append(3)
        return valid_actions

    def step(self, action):
        reward = 0
        winnings = 0
        has_doubled = False
        player_has_busted = False
        blackjack = False

        if not self.check_dealer_blackjack():
            if action == 0:  # Hit
                self.player_hand.add_card(self.shoe.deal(is_dealer=False, length_of_hand=len(self.player_hand.cards)))
                if self.player_hand.hand_value() > 21:  # Player busts
                    player_has_busted = True
                    reward = -1
                    winnings = -1
                    self.done = True

            elif action == 1:  # Stand
                self.done = True

            elif action == 2:  # Double
                if len(self.player_hand.cards) == 2: 
                    self.player_hand.has_doubled = True
                    self.player_hand.add_card(self.shoe.deal(is_dealer=False, length_of_hand=len(self.player_hand.cards)))
                    has_doubled = True
                    if self.player_hand.hand_value() > 21:
                        player_has_busted = True
                        reward = -2
                        winnings = -2
                        self.done = True
                    else:
                        self.done = True
                else:
                    raise ValueError("AI is trying to double a hand it is not supposed to")

            elif action == 3:
                if len(self.player_hand.cards) == 2:
                    self.player_hand.has_split = True
                    self.player_hand.split_multiplier += 1
                    self.player_hand.add_card(self.shoe.deal(is_dealer=False,length_of_hand=self.player_hand.split_multiplier))
                    self.player_hand.cards.pop(0)
                    self.player_hand.hand_value()
                    self.get_valid_actions()
                else:
                    raise ValueError("AI tried to split an unqualified hand")

            if self.done:
                player_value = self.player_hand.hand_value()
                dealer_value = self.dealer_hand.hand_value()

                if self.check_player_blackjack():
                    blackjack = True

                while dealer_value < 17 and not player_has_busted:
                    self.dealer_hand.add_card(self.shoe.deal(is_dealer=True, length_of_hand=len(self.dealer_hand.cards)))
                    dealer_value = self.dealer_hand.hand_value()

                if not player_has_busted:
                    if blackjack:
                        reward = 1.5
                        winnings = 1.5
                    elif dealer_value > 21 or (player_value > dealer_value and not player_has_busted):
                        reward = 2 if has_doubled else 1
                        winnings = 2 if has_doubled else 1
                    elif player_value < dealer_value:
                        reward = -2 if has_doubled else -1
                        winnings = -2 if has_doubled else -1
                    elif player_value == dealer_value:
                        reward = 0
                        winnings = 0
                reward *= self.player_hand.split_multiplier
                winnings *= self.player_hand.split_multiplier
        else:
            if self.check_player_blackjack():
                reward = 0
                winnings = 0
            else:
                reward = 0 
                winnings = -1
            self.done = True

        if self.is_hand_over():
            self.shoe.deal(hand_over_signal=True)

        next_state = self.get_state_representation()
        return next_state, reward, winnings, self.done


class DQN(nn.Module):
    def __init__(self, action_size=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, LAYER_1_SIZE)
        self.fc2 = nn.Linear(LAYER_1_SIZE, LAYER_2_SIZE)
        self.fc3 = nn.Linear(LAYER_2_SIZE, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, env, batch_size, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate, run_number, action_size=4, state_size=4):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.run_number = run_number

        self.model = DQN(action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-6)
        self.criterion = nn.MSELoss()


    def train(self, episodes, log_file, process_callback=None):
        rewards_list = []
        winnings_list = []
        cumulative_reward = 0
        cumulative_winnings = 0
        log_buffer = []
        log_interval = max(1, episodes // 20)
        open(log_file, "w").close()

        for episode in range(episodes):
            state = self.env.reset()
            state = state.to(device)
            if isinstance(state, torch.Tensor):
                state = state.clone().detach()
            else:
                state = torch.tensor(state, dtype=torch.float32).to(device)

            total_reward = 0
            total_winnings = 0
            while True:
                valid_actions = self.env.get_valid_actions()
                if random.random() < self.epsilon:
                    action = random.choice(valid_actions)
                else:
                    with torch.no_grad():
                        q_values = self.model(state)
                        masked_q_values = torch.full_like(q_values, -float('inf')).to(device)
                        for valid_action in valid_actions:
                            masked_q_values[valid_action] = q_values[valid_action]
                        action = masked_q_values.argmax().item()

                next_state, reward, winnings, done = self.env.step(action)


                log_buffer.append(log_decision(state, valid_actions, action, reward, winnings, done))


                if isinstance(next_state, torch.Tensor):
                    next_state = next_state.clone().detach()
                else:
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                total_winnings += winnings

                if done:
                    break

            if len(self.memory) >= self.batch_size:
                self.replay()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            cumulative_reward += total_reward
            cumulative_winnings += total_winnings
            rewards_list.append(cumulative_reward)
            winnings_list.append(cumulative_winnings)

            if (episode + 1) % log_interval == 0 or (episode +1) == episodes:
                with open(log_file, "a") as file:
                    for entry in log_buffer:
                        file.write(json.dumps(entry) + "\n")
                log_buffer.clear()

            if process_callback and (episode +1) % 500 == 0:
                process_callback(episode + 1)

        self.plot_data(rewards_list, data_type='rewards')
        self.plot_data(winnings_list, data_type='winnings')

    def plot_data(self, data_list, data_type ):
        plt.figure()  
        plt.plot(data_list)

        if data_type == 'rewards':
            plt.title('Rewards')
            filename = f'rewards_plot{self.run_number}.png'
        else:
            plt.title('Winnings')
            filename = f'winnings_plot{self.run_number}.png'

        plt.xlabel('Episode')
        plt.ylabel('Cumulative Value')

        save_path = os.path.join('../outputs/plots', filename)
        os.makedirs('plots', exist_ok=True)  
        plt.savefig(save_path)
        plt.close() 

    def replay(self):
        indices = random.sample(range(len(self.memory)), self.batch_size)
        batch = [self.memory[i] for i in indices] 

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(device)
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values = next_q_values.max(1, keepdim=True)[0]
            targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        q_values = self.model(states).gather(1, actions)
        loss = self.criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def load_logs(filename=INPUT_HANDS):
    with open(filename, "r") as file:
        logs = json.load(file)

    dealer_cards = [[int(v) if isinstance(v, (int, str)) else v.value for v in hand] for hand in logs["dealer_cards"]]
    player_cards = [[int(v) if isinstance(v, (int, str)) else v.value for v in hand] for hand in logs["player_cards"]]

    return dealer_cards, player_cards

def log_decision(state, valid_actions, chosen_action, reward, winnings, done):
    return {
        "state": state.cpu().tolist() if isinstance(state, torch.Tensor) else state,
        "valid_actions": valid_actions,
        "chosen_action": chosen_action,
        "reward": reward,
        "winnings": winnings,
        "done": done,
    }

def train_model_wrapper(param_index, param_dict, dealer_hands, player_hands):
    desc = f"Model {param_index+1}"
    total_steps = len(dealer_hands) 
    pbar = tqdm(total=total_steps, desc=desc, position=param_index, leave=True)

    env = BlackjackEnvAI(dealer_sequence=dealer_hands, player_sequence=player_hands)

    agent = DQNAgent(
        env,
        run_number=param_index + 1,
        batch_size=param_dict["batch_size"],
        gamma=param_dict["gamma"],
        epsilon=1,
        learning_rate=param_dict["learning_rate"],
        epsilon_decay=param_dict["epsilon_decay"],
        epsilon_min=param_dict["epsilon_min"]
    )

    log_file = f"{LOG_OUTPUT}/training_log_model_{param_index+1}.json"

    def progress_callback(current_step):
            pbar.n = current_step
            pbar.refresh()

    agent.train(total_steps, log_file, process_callback=progress_callback)
    pbar.close()

    model_path = f"{AGENT_OUTPUT}/model_{param_index+1}.pth"
    torch.save(agent.model.state_dict(), model_path)
    print(f"Model {param_index+1} saved at {model_path}")



if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    dealer_hands, player_hands = load_logs(INPUT_HANDS)

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, param in enumerate(parameter_sets):
            futures.append(
                executor.submit(train_model_wrapper, i, param, dealer_hands, player_hands)
            )

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Training failed with error: {e}")
