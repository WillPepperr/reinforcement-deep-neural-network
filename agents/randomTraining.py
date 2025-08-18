import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import cProfile
import pstats


PERFORMANCE_DEBUG = False


class Card:
    values = {
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        'T': 10, 'J': 10, 'Q': 10, 'K': 10,
        'A': 11
    }

    def __init__(self, value):
        self.value = Card.values[value]

    def __repr__(self):
        return f"{self.value}"


class Shoe:
    def __init__(self, num_decks, fixed_dealer_sequence=None, fixed_player_sequence=None):
        self.all_cards = num_decks * 52
        self.decks = num_decks
        self.shoe_cards = []
        self.create_shoe()

    def create_shoe(self):
        self.shoe_cards = [Card(v) for _ in range(self.decks) for _ in range(4) for v in Card.values]
        self.shuffle_shoe()

    def shuffle_shoe(self):
        random.shuffle(self.shoe_cards)

    def deal(self):
        card = self.shoe_cards.pop()
        return card


class Hand:
    def __init__(self):
        self.cards = []
        self.has_doubled = False
        self.has_split = False
        self.num_aces = 0
        self.is_soft = False

    def add_card(self, card):
        self.cards.append(card)
        self.num_aces = sum(1 for card in self.cards if card.value == 11)

    def check_soft(self):
        non_ace_sum = 0
        for card in self.cards:
            if card.value == 11:
                non_ace_sum += 1
            else:
                non_ace_sum += card.value
        if self.num_aces > 0 and non_ace_sum <= 11:
            self.is_soft = True
        else:
            self.is_soft = False

    def hand_value(self):
        value = sum(card.value for card in self.cards)
        num_aces = self.num_aces
        while value > 21 and num_aces > 0:
            value -= 10
            num_aces -= 1
        return value

class BlackjackEnvAI:
    def __init__(self, num_decks=8):
        self.done = False
        self.shoe = Shoe(num_decks)
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.reset()

    def is_hand_over(self):
        if self.done:
            return True
        return False

    def reset(self):
        self.shoe.create_shoe()
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.player_hand.add_card(self.shoe.deal())
        self.player_hand.add_card(self.shoe.deal())
        self.dealer_hand.add_card(self.shoe.deal())
        self.dealer_hand.add_card(self.shoe.deal())
        self.done = False

        return self.get_state_representation()

    def get_state_representation(self):
        player_total = self.player_hand.hand_value()
        is_soft = 1 if self.player_hand.check_soft() else 0
        dealer_card_value = self.dealer_hand.cards[0].value if len(self.dealer_hand.cards) > 0 else 0
        has_doubled = 1 if self.player_hand.has_doubled else 0
        has_split = 1 if self.player_hand.has_split else 0

        state_representation = [
            player_total,
            is_soft,
            dealer_card_value,
            has_doubled,
            has_split
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
        if len(self.player_hand.cards) == 2 and self.player_hand.cards[0].value == self.player_hand.cards[1].value:
            valid_actions.append(3)
        return valid_actions

    def step(self, action):
        reward = 0
        has_doubled = False
        has_busted = False
        blackjack = False

        if not self.check_dealer_blackjack():
            if action == 0:  # Hit
                self.player_hand.add_card(self.shoe.deal())
                if self.player_hand.hand_value() > 21:  # Player busts
                    has_busted = True
                    reward = -1
                    self.done = True

            elif action == 1:  # Stand
                self.done = True

            elif action == 2:  # Double
                if len(self.player_hand.cards) == 2:
                    self.player_hand.has_doubled = True
                    self.player_hand.add_card(self.shoe.deal())
                    has_doubled = True
                    if self.player_hand.hand_value() > 21:
                        has_busted = True
                        reward = -2
                        self.done = True
                    else:
                        self.done = True
                else:
                    raise ValueError("Double action is only allowed on the first two cards.")

            elif action == 3:
                self.player_hand.cards.pop(1)
                self.player_hand.add_card(self.shoe.deal())
                self.player_hand.has_split = True

            if self.done:
                player_value = self.player_hand.hand_value()
                dealer_value = self.dealer_hand.hand_value()
                if self.check_player_blackjack():
                    blackjack = True
                while self.dealer_hand.hand_value() < 17 and has_busted is False:
                    self.dealer_hand.add_card(self.shoe.deal())
                    dealer_value = self.dealer_hand.hand_value()

                if blackjack is True:
                    reward = 2.5

                elif dealer_value > 21 or (player_value > dealer_value and has_busted is False):
                    if has_doubled and self.player_hand.has_split:
                        reward = 4
                    elif has_doubled ^ self.player_hand.has_split:
                        reward = 2
                    else:
                        reward = 1

                elif player_value < dealer_value:  # Dealer wins
                    if has_doubled and self.player_hand.has_split:
                        reward = -4
                    elif has_doubled ^ self.player_hand.has_split:
                        reward = -2
                    else:
                        reward = -1
                elif player_value == dealer_value:
                    reward = 0

        else:
            if self.check_player_blackjack():
                reward = 0
            else:
                reward = -1
            self.done = True

        next_state = self.get_state_representation()
        return next_state, reward, self.done


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, action_size=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Agent class with training logic
class DQNAgent:
    def __init__(self, env, action_size=4, state_size=5, batch_size=128, gamma=0.30, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.0005, learn_rate=0.005):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model = DQN(action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)
        self.criterion = nn.MSELoss()

    def train(self, episodes=1000000):
        rewards_list = []
        cumulative_reward = 0
        for episode in tqdm(range(episodes), desc="Training Progress", miniters=250):
            state = self.env.reset()
            state = state.to(device)
            if isinstance(state, torch.Tensor):
                state = state.clone().detach()
            else:
                state = torch.tensor(state, dtype=torch.float32).to(device)

            total_reward = 0

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

                next_state, reward, done = self.env.step(action)
                if isinstance(next_state, torch.Tensor):
                    next_state = next_state.clone().detach()
                else:
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if done:
                    break

            # Train the model if memory has enough samples
            if len(self.memory) >= self.batch_size:
                self.replay()

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            cumulative_reward += total_reward
            rewards_list.append(cumulative_reward)

        self.plot_rewards(rewards_list)


    def plot_rewards(self, rewards_list):
        plt.plot(rewards_list)
        plt.title('Cumulative Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.show()

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
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





if __name__ == "__main__":
    environment = BlackjackEnvAI()
    agent = DQNAgent(environment)

    if PERFORMANCE_DEBUG:
        profiler = cProfile.Profile()
        with profiler:
            agent.train()

        stats = pstats.Stats(profiler)

        stats.strip_dirs()
        stats.sort_stats("tottime")
        stats.print_stats(75)

    else:
        agent.train()

    torch.save(agent.model.state_dict(), 'dqn_model.pth')
