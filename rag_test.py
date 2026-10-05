import re

with open("ai_knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Step 1: Split document into sentences
sentences = re.split(r'(?<=[.!?])\s+', text)

# Step 2: Chunk settings
chunk_size = 200
chunk_overlap = 1

# Step 3: Create chunks
chunks = []
current_chunk = []

for sentence in sentences:

    test_chunk = current_chunk + [sentence]
    test_text = " ".join(test_chunk)

    if len(test_text) <= chunk_size:
        current_chunk = test_chunk

    else:
        chunks.append(" ".join(current_chunk))

        overlap_sentences = current_chunk[-chunk_overlap:]

        current_chunk = overlap_sentences + [sentence]

# Step 4: Save the final chunk
if current_chunk:
    chunks.append(" ".join(current_chunk))

# Step 5: Display the chunks
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print("\nChunk", i + 1)
    print(chunk)