import pandas as pd
from kafka import KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
import json
import time
import os


consumer = KafkaConsumer(
    'fintech',
    bootstrap_servers='kafka:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda x:json.loads(x.decode('utf-8')),
    consumer_timeout_ms=2000
)







 
