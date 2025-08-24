from bjcards import *
import torch
import torch.nn as nn
import json

INPUT_HANDS = "../data/ten_million.json"
NN_PATH = "../outputs/saved_nns/model_4.pth"
LAYER_1 = 16
LAYER_2 = 16

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
                if current_hand:  # Ensure the hand is not empty
                    card = current_hand[length_of_hand]
        if is_dealer is False:
            if self.player_index <= len(self.fixed_player_sequence):
                current_hand = self.fixed_player_sequence[self.player_index]
                if current_hand:  # Ensure the hand is not empty
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
                    raise ValueError("ai is trying to double a hand it is not supposed to")

            elif action == 3:
                if len(self.player_hand.cards) == 2:
                    self.player_hand.has_split = True
                    self.player_hand.split_multiplier += 1
                    self.player_hand.add_card(self.shoe.deal(is_dealer=False,length_of_hand=self.player_hand.split_multiplier))
                    self.player_hand.cards.pop(0)
                    self.player_hand.hand_value()
                    self.get_valid_actions()
                else:
                    raise ValueError("Ai tried to split an unqualified hand")

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
            # Handle dealer blackjack cases
            if self.check_player_blackjack():
                reward = 0
                winnings = 0
            else:
                reward = 0 
                winnings = -1
            self.done = True

        # Signal hand completion to Shoe
        if self.is_hand_over():
            self.shoe.deal(hand_over_signal=True)

        next_state = self.get_state_representation()
        return next_state, reward, winnings, self.done



class DQN(nn.Module):
    def __init__(self, action_size=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, LAYER_1)
        self.fc2 = nn.Linear(LAYER_1, LAYER_2)
        self.fc3 = nn.Linear(LAYER_2, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)



def load_logs(filename="card_logs.json"):
    with open(filename, "r") as file:
        logs = json.load(file)

     # Ensure only integers are passed as card values
    dealer_cards = [[int(v) if isinstance(v, (int, str)) else v.value for v in hand] for hand in logs["dealer_cards"]]
    player_cards = [[int(v) if isinstance(v, (int, str)) else v.value for v in hand] for hand in logs["player_cards"]]

    return dealer_cards, player_cards

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing device: {device}")

    dealer_hands, player_hands = load_logs(INPUT_HANDS)

    env = BlackjackEnvAI(dealer_sequence=dealer_hands, player_sequence=player_hands)

    model = DQN(action_size=4).to(device)
    for i in range(4):
        NN_PATH = f"../outputs/saved_nns/model_{i+1}.pth"
        model.load_state_dict(torch.load(NN_PATH, map_location=device))
        model.eval()

        total_reward = 0
        total_winnings = 0
        for episode in range(len(dealer_hands)):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_winnings = 0
            if episode % 10000 == 0:
                print(f"{episode} completed episodes")
            while not done:
                state = state.to(device)
                with torch.no_grad():
                    q_values = model(state)
                    valid_actions = env.get_valid_actions()
                    masked_q_values = torch.full_like(q_values, -float('inf'))
                    for a in valid_actions:
                        masked_q_values[a] = q_values[a]
                    action = masked_q_values.argmax().item()
                state, reward, winnings, done = env.step(action)
                episode_reward += reward
                episode_winnings += winnings
            total_reward += episode_reward
            total_winnings += episode_winnings 
        print(f"Average reward: {total_reward / len(dealer_hands)}")
        print(f"Average winnings: {total_winnings / len(dealer_hands)}")
