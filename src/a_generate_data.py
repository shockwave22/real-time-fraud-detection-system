from faker import Faker
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

fake = Faker()

def generate_transactions(num_records=10000):
    data = []
    for _ in range(num_records):
        # Generate legitimate transaction
        transaction = {
            'transaction_id': fake.uuid4(),
            'user_id': fake.random_int(1000, 9999),
            'amount': round(np.random.lognormal(3, 1.2), 2),
            'merchant': fake.company(),
            'category': random.choice(['Groceries', 'Gas', 'Online', 'Retail', 'Dining']),
            'location': fake.country_code(),
            'device': random.choice(['Mobile', 'Desktop', 'Tablet']),
            'time': fake.date_time_between('-30d', 'now').isoformat(),
            'ip_address': fake.ipv4()
        }
        
        # Introduce fraud patterns (3% fraud rate)
        is_fraud = 0
        if random.random() < 0.03:
            is_fraud = 1
            transaction['amount'] *= random.uniform(5, 15)  # 5-15x normal amount
            if random.random() > 0.5:
                transaction['location'] = fake.country_code()  # Foreign transaction
            transaction['time'] = (datetime.fromisoformat(transaction['time']) + 
                                  timedelta(minutes=random.randint(1, 10))).isoformat()
        
        transaction['is_fraud'] = is_fraud
        data.append(transaction)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_transactions(20000)
    df.to_csv('E:\Projects\Credit Card Fraud detection\data\generated_data.csv', index=False)
    print("Generated 20,000 transactions with fraud patterns")