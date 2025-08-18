import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
import cProfile
import pstats
import json


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
        self.fixed_dealer_sequence = [[Card(v) for v in hand] for hand in (fixed_dealer_sequence or [])]
        self.fixed_player_sequence = [[Card(v) for v in hand] for hand in (fixed_player_sequence or [])]
        self.player_card_log = []
        self.dealer_card_log = []
        self.dealer_index = 0
        self.player_index = 0
        if not fixed_dealer_sequence and fixed_player_sequence:
            self.create_shoe()

    def create_shoe(self):
        self.shoe_cards = [Card(v) for _ in range(self.decks) for _ in range(4) for v in Card.values]
        self.shuffle_shoe()

    def shuffle_shoe(self):
        random.shuffle(self.shoe_cards)

    def deal(self, is_dealer=False):
        if is_dealer:
            if self.fixed_dealer_sequence and self.dealer_index < len(self.fixed_dealer_sequence):
                current_hand = self.fixed_dealer_sequence[self.dealer_index]
                if len(current_hand) > 0:
                    card = current_hand.pop(0)
                else:
                    self.dealer_index += 1  # Move to the next hand
                    card = self.shoe_cards.pop()
            else:
                card = self.shoe_cards.pop()
            self.dealer_card_log.append(card)
        else:
            if self.fixed_player_sequence and self.player_index < len(self.fixed_player_sequence):
                current_hand = self.fixed_player_sequence[self.player_index]
                if len(current_hand) > 0:
                    card = current_hand.pop(0)
                else:
                    self.player_index += 1  # Move to the next hand
                    card = self.shoe_cards.pop()
            else:
                card = self.shoe_cards.pop()
            self.player_card_log.append(card)
        return card


class Hand:
    def __init__(self):
        self.cards = []
        self.has_split = False
        self.split_multiplier = 1
        self.has_doubled = False
        self.num_aces = 0
        self.is_soft = None 
        self.check_soft()

    def add_card(self, card) -> None:
        self.cards.append(card)
        self.num_aces = sum(1 for card in self.cards if card.value == 11)

    def hand_value(self) -> int:
        value = sum(card.value for card in self.cards)
        num_aces = self.num_aces
        while value > 21 and num_aces > 0:
            value -= 10
            num_aces -= 1
        return value

    def check_soft(self) -> None:
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
        
    def hand_is_splitable(self) -> bool:
        if self.cards[0].value == self.cards[1].value:
            if len(self.cards) == 2:
                return True
        return False


class BlackjackEnvAI:
    def __init__(self, num_decks=8):
        self.done = False
        self.shoe = Shoe(num_decks)
        self.dealer_card_log = []
        self.player_card_log = []
        self.reset()

    def is_hand_over(self) -> bool:
        if self.done:
            return True
        return False

    def reset(self): 
        self.shoe.create_shoe()
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.ghost_cards = []
        self.player_hand.add_card(self.shoe.deal())
        self.player_hand.add_card(self.shoe.deal())
        self.dealer_hand.add_card(self.shoe.deal())
        self.dealer_hand.add_card(self.shoe.deal())
        self.done = False

        return self.get_state_representation()

    def get_state_representation(self):
        player_total = self.player_hand.hand_value()
        is_soft = 1 if self.player_hand.num_aces > 0 and player_total <= 21 else 0
        dealer_card_value = self.dealer_hand.cards[0].value if len(self.dealer_hand.cards) > 0 else 0
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
            valid_actions = [3] 
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
                if len(self.player_hand.cards) == 2:  # Double is only valid on first two cards
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
                if len(self.player_hand.cards) == 2:
                    self.player_hand.has_split = True
                    self.player_hand.split_multiplier += 1
                    self.ghost_cards.append(self.player_hand.cards[0])
                    self.player_hand.cards.pop(0)
                    self.player_hand.add_card(self.shoe.deal())
                else:
                    raise ValueError("Ai tried to split an unqualified hand")

            if self.done:
                player_value = self.player_hand.hand_value()
                dealer_value = self.dealer_hand.hand_value()
                if self.check_player_blackjack():
                    self.dealer_hand.add_card(self.shoe.deal())
                    if player_value != self.dealer_hand.hand_value():
                        blackjack = True
                while self.dealer_hand.hand_value() < 17 and has_busted is False:
                    self.dealer_hand.add_card(self.shoe.deal())
                    dealer_value = self.dealer_hand.hand_value()
                if blackjack is True:
                    reward = 1.5
                elif dealer_value > 21 or (player_value > dealer_value and has_busted is False):
                    reward = 2 if has_doubled else 1
                elif player_value < dealer_value:  # Dealer wins
                    reward = -2 if has_doubled else -1
                elif player_value == dealer_value:
                    reward = 0
                reward *= self.player_hand.split_multiplier

                while player_value < 31 and not self.player_hand.check_soft():
                    self.player_hand.add_card(self.shoe.deal())
                    player_value = self.player_hand.hand_value()

                while dealer_value < 21 and not self.dealer_hand.check_soft():
                    self.dealer_hand.add_card(self.shoe.deal())
                    dealer_value = self.dealer_hand.hand_value()
                    
                    
                ghost_values = [int(card.value) for card in self.ghost_cards]
                player_values = [int(card.value) for card in self.player_hand.cards]

                self.player_card_log.append(ghost_values + player_values)
                self.dealer_card_log.append([int(card.value) for card in self.dealer_hand.cards])

        else:
            if self.check_player_blackjack():
                reward = 0
            else:
                reward = -1
            self.done = True

            self.player_card_log.append([int(card.value) for card in self.player_hand.cards])            
            self.dealer_card_log.append([int(card.value) for card in self.dealer_hand.cards])

        next_state = self.get_state_representation()
        return next_state, reward, self.done




class DQN(nn.Module):
    def __init__(self, action_size=3):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, env, action_size=4, state_size=4, batch_size=128, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model = DQN(action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.MSELoss()

    def train(self, episodes=131072):
        rewards_list = []
        cumulative_reward = 0
        for _ in tqdm(range(episodes), desc= "Training Progress", miniters=250):
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


def save_logs(dealer_log, player_log, filename="test_json/unnamed_result"):
    logs = {
        "dealer_cards": dealer_log,
        "player_cards": player_log
    }

    with open(filename, "w") as file:
        file.write("{\n")

        # Write dealer_cards with each inner list on a new line
        file.write('  "dealer_cards": [\n')
        for i, dealer_hand in enumerate(dealer_log):
            file.write(f"    {json.dumps(dealer_hand)}")
            if i < len(dealer_log) - 1:
                file.write(",")
            file.write("\n")
        file.write("  ],\n")

        # Write player_cards with each inner list on a new line
        file.write('  "player_cards": [\n')
        for i, player_hand in enumerate(player_log):
            file.write(f"    {json.dumps(player_hand)}")
            if i < len(player_log) - 1:
                file.write(",")
            file.write("\n")
        file.write("  ]\n")

        file.write("}\n")


def load_logs(filename="card_logs.json"):
    with open(filename, "r") as file:
        logs = json.load(file)
    return logs["dealer_cards"], logs["player_cards"]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(5):
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

        save_logs(environment.dealer_card_log, environment.player_card_log, f"test_json/100k({i + 5}).json")
        print(f"finished {i +1} training loops")
