class Card:
    def __init__(self, value):
        if not isinstance(value, int):
            raise ValueError(f"Card value must be an integer, got {type(value).__name__}: {value}")
        self.value = value

    def __repr__(self):
        return f"Card({self.value})"

class Hand:
    def __init__(self):
        self.cards = []
        self.has_doubled = False
        self.has_split = False
        self.split_multiplier = 1
        self.num_aces = 0
        self.is_soft = False

    def add_card(self, card):
        if not isinstance(card, Card):
            raise TypeError(f"Attempted to add an invalid card: {card}")
        self.cards.append(card)
        self.num_aces = sum(1 for card in self.cards if card.value == 11)

    def hand_value(self) -> int:
        value = sum(card.value for card in self.cards)
        num_aces = self.num_aces
        while value > 21 and num_aces > 0:
            value -= 10
            num_aces -= 1
        if num_aces > 0:
            self.is_soft = True
        else:
            self.is_soft = False
        return value
    
    def hand_is_splitable(self) -> bool:
        if self.cards[0].value == self.cards[1].value:
            if len(self.cards) == 2:
                return True
        return False 
