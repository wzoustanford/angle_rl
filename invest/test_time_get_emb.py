import time
from utils import get_news_embedding

t1 = time.time()
emb = get_news_embedding('AAPL', '2025-03-05', '2025-04-05')
t2 = time.time() 

#print(emb)
print(f"elapsed time: {t2-t1} seconds")
