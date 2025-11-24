import json


# looking for mismatches in rewards to debug why bot isnt recieveing dealer blackjacks
file_path = '../nn_decision_logs.json' 

count = 0
    
with open(file_path, 'r') as file:
    for line_number, line in enumerate(file, start=1):
        try:
            data = json.loads(line)
            if data.get('winnings') != data.get('reward'):
                count += 1
        except json.JSONDecodeError:
                continue 
print(f"Number of mismatches: {count}")
                
