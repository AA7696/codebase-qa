from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I send an HTTP GET request?",
    "Function that performs a GET request to a URL",
    "Recipe for chocolate cake",
]

vectors = model.encode(sentences)
print("Each vector has", len(vectors[0]), "numbers")
print("First 5 numbers of sentence 1:", vectors[0][:5])

print("Sentence 1 vs 2:", float(util.cos_sim(vectors[0], vectors[1])))
print("Sentence 1 vs 3:", float(util.cos_sim(vectors[0], vectors[2])))