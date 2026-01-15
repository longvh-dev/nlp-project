import os
from collections import Counter
from typing import List

from src.data.conllu_parser import parse_conllu, Sentence, Token


def analyze_dataset(file_path: str, name: str):
    print(f"\n--- Analyzing {name} dataset: {file_path} ---")
    sentences: List[Sentence] = parse_conllu(file_path)

    if not sentences:
        print("No sentences parsed.")
        return

    num_sentences = len(sentences)
    num_tokens = sum(len(s.tokens) for s in sentences)

    upos_tags = Counter()
    deprel_tags = Counter()
    sentence_lengths = []

    for sentence in sentences:
        sentence_lengths.append(len(sentence.tokens))
        for token in sentence.tokens:
            if token.upos != "_":
                upos_tags[token.upos] += 1
            if token.deprel != "_":
                deprel_tags[token.deprel] += 1

    print(f"Number of sentences: {num_sentences}")
    print(f"Number of tokens: {num_tokens}")

    print("\nTop 10 Universal Part-of-Speech (UPOS) Tags:")
    for tag, count in upos_tags.most_common(10):
        print(f"  {tag}: {count} ({count / num_tokens:.2%})")

    print("\nTop 10 Dependency Relations (DEPREL):")
    for rel, count in deprel_tags.most_common(10):
        print(f"  {rel}: {count} ({count / num_tokens:.2%})")

    avg_sentence_length = sum(sentence_lengths) / num_sentences
    max_sentence_length = max(sentence_lengths)
    min_sentence_length = min(sentence_lengths)

    print(f"\nSentence Lengths:")
    print(f"  Average: {avg_sentence_length:.2f} tokens")
    print(f"  Max: {max_sentence_length} tokens")
    print(f"  Min: {min_sentence_length} tokens")


if __name__ == "__main__":
    data_dir = "data"
    train_file = os.path.join(data_dir, "vi_vtb-ud-train.conllu")
    dev_file = os.path.join(data_dir, "vi_vtb-ud-dev.conllu")
    test_file = os.path.join(data_dir, "vi_vtb-ud-test.conllu")

    analyze_dataset(train_file, "Training")
    analyze_dataset(dev_file, "Development")
    analyze_dataset(test_file, "Test")
