import json



def load_logs(filename="card_logs.json"):
    with open(filename, "r") as file:
        logs = json.load(file)

    # Ensure only integers are passed as card values
    dealer_cards = [[int(v) if isinstance(v, (int, str)) else v.value for v in hand] for hand in logs["dealer_cards"]]
    player_cards = [[int(v) if isinstance(v, (int, str)) else v.value for v in hand] for hand in logs["player_cards"]]

    return dealer_cards, player_cards

class Card:
    def __init__(self, value):
        if not isinstance(value, int):
            raise ValueError(f"Card value must be an integer, got {type(value).__name__}: {value}")
        self.value = value

    def __repr__(self):
        return f"Card({self.value})"

class Shoe:
    def __init__(self, fixed_dealer_sequence=None, fixed_player_sequence=None):
        self.fixed_dealer_sequence = [[Card(v.value if isinstance(v, Card) else v) for v in hand] for hand in (fixed_dealer_sequence or [])]
        self.fixed_player_sequence = [[Card(v.value if isinstance(v, Card) else v) for v in hand]for hand in (fixed_player_sequence or [])]
        self.dealer_index = 0
        self.player_index = 0

    def deal(self, is_dealer=False, hand_over_signal=False):
        card = None  # Initialize to None

        if hand_over_signal:
            if is_dealer:
                self.dealer_index += 1
            else:
                self.player_index += 1

        if is_dealer:
            if self.fixed_dealer_sequence and self.dealer_index < len(self.fixed_dealer_sequence):
                current_hand = self.fixed_dealer_sequence[self.dealer_index]
                if len(current_hand) > 0:
                    card = current_hand.pop(0)
        else:
            if self.fixed_player_sequence and self.player_index < len(self.fixed_player_sequence):
                current_hand = self.fixed_player_sequence[self.player_index]
                if len(current_hand) > 0:
                    card = current_hand.pop(0)

        # Validate that the returned card is not None
        if card is None:
            raise TypeError(
                f"Invalid card dealt: {card}. Ensure only Card objects are returned."
                f"\nDealer index: {self.dealer_index}, Player index: {self.player_index}"
                f"\nDealer sequence: {self.fixed_dealer_sequence}"
                f"\nPlayer sequence: {self.fixed_player_sequence}"
            )
        return card


if __name__ == "__main__":
    dealer_hands, player_hands = load_logs("test_json/card_logs.json")
    shoe = Shoe(fixed_dealer_sequence=dealer_hands, fixed_player_sequence=player_hands)
    print(shoe.fixed_dealer_sequence[1011])


