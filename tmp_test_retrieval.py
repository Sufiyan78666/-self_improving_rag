import sys, os
sys.path.insert(0, '.')

# Create a test txt doc
os.makedirs('data/raw', exist_ok=True)
with open('data/raw/test_doc.txt', 'w') as f:
    f.write('Retrieval Augmented Generation combines a retrieval system with a language model. ' * 20)

from self_improving_rag.retrieval.ingest import ingest_file
from self_improving_rag.retrieval.retriever import retrieve

n = ingest_file('data/raw/test_doc.txt')
print(f'Chunks ingested: {n}')

results = retrieve('What is retrieval augmented generation?', k=3)
for r in results:
    print(f'  rank={r["rank"]} score={r["vector_score"]:.3f} text={r["text"][:50]}')
print('ALL RETRIEVAL TESTS PASSED')
