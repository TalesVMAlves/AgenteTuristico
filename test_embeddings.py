import numpy as np
import sys
from get_embedding_function import get_embedding_function
from langchain.evaluation import load_evaluator

def test_embedding_function(word1, word2, word3):
    # Get the embedding function.
    embedding_function = get_embedding_function()

    embedding_1 = embedding_function.embed_query(word1)
    print(f"Texto: {word1}, Tamanho do Vetor: {len(embedding_1)}\nEmbedding: {embedding_1[:5]} ...")
    embedding_2 = embedding_function.embed_query(word2)
    print(f"Texto: {word2}, Tamanho do Vetor: {len(embedding_2)}\nEmbedding: {embedding_2[:5]} ...")
    embedding_3 = embedding_function.embed_query(word3)
    print(f"Texto: {word3}, Tamanho do Vetor: {len(embedding_3)}\nEmbedding: {embedding_3[:5]} ...")

    # Calculate cosine similarities between each pair of embeddings
    similarity_word1_word2 = np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
    print(f"Similaridade entre '{word1}' e '{word2}': {similarity_word1_word2:.4f}")

    similarity_word1_word3 = np.dot(embedding_1, embedding_3) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_3))
    print(f"Similaridade entre '{word1}' e '{word3}': {similarity_word1_word3:.4f}")

    similarity_word2_word3 = np.dot(embedding_2, embedding_3) / (np.linalg.norm(embedding_2) * np.linalg.norm(embedding_3))
    print(f"Similaridade entre '{word2}' e '{word3}': {similarity_word2_word3:.4f}")

if __name__ == "__main__":
    # Check if the correct number of arguments are passed
    if len(sys.argv) != 4:
        print("Usage: python test_embeddings.py <word1> <word2> <word3>")
        sys.exit(1)

    # Capture words from command-line arguments
    word1 = sys.argv[1]
    word2 = sys.argv[2]
    word3 = sys.argv[3]
    
    # Call the test function with the command-line arguments
    test_embedding_function(word1, word2, word3)
# A simple Fibonacci function that calculates the nth Fibonacci number

def fibonacci(n):
    if n <= 0:
        return "Input should be a positive integer."
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        fib_sequence = [0, 1]
        while len(fib_sequence) < n:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        return fib_sequence[-1]

# Test the Fibonacci function
print("Fibonacci(5) =", fibonacci(5))
