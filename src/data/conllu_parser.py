from typing import List, Dict, Any

class Token:
    def __init__(self, fields: List[str]):
        self.id = fields[0]
        self.form = fields[1]
        self.lemma = fields[2]
        self.upos = fields[3]
        self.xpos = fields[4]
        self.feats = fields[5]
        self.head = fields[6]
        self.deprel = fields[7]
        self.deps = fields[8]
        self.misc = fields[9]

    def __repr__(self):
        return f"Token(id={self.id}, form='{self.form}', upos='{self.upos}', deprel='{self.deprel}')"

class Sentence:
    def __init__(self, tokens: List[Token], metadata: Dict[str, str]):
        self.tokens = tokens
        self.metadata = metadata

    def __repr__(self):
        text = self.metadata.get('text', 'N/A')
        return f"Sentence(text='{text}', num_tokens={len(self.tokens)})"

def parse_conllu(file_path: str) -> List[Sentence]:
    sentences = []
    current_tokens: List[Token] = []
    current_metadata: Dict[str, str] = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Empty line indicates end of sentence
                    if current_tokens:
                        sentences.append(Sentence(current_tokens, current_metadata))
                    current_tokens = []
                    current_metadata = {}
                elif line.startswith('#'):  # Metadata line
                    parts = line[1:].strip().split('=', 1)
                    if len(parts) == 2:
                        current_metadata[parts[0].strip()] = parts[1].strip()
                else:  # Token line
                    fields = line.split('\t')
                    if len(fields) == 10:
                        current_tokens.append(Token(fields))
                    else:
                        # Handle potential multi-word tokens or malformed lines, for now just skip
                        pass
            # Add the last sentence if the file doesn't end with an empty line
            if current_tokens:
                sentences.append(Sentence(current_tokens, current_metadata))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error parsing CoNLL-U file: {e}")

    return sentences

if __name__ == '__main__':
    # Example usage:
    # Assuming a dummy.conllu file exists for testing
    dummy_conllu_content = """
# sent_id = test-s1
# text =This is a test sentence.
1       This    this    DET     DT      _       3       nsubj   _       _
2       is      be      AUX     VBZ     _       3       cop     _       _
3       a       a       DET     DT      _       5       det     _       _
4       test    test    NOUN    NN      _       5       compound        _       _
5       sentence        sentence        NOUN    NN      _       0       root    _       _
6       .       .       PUNCT   .       _       5       punct   _       _

# sent_id = test-s2
# text =Another one.
1       Another another DET     DT      _       2       det     _       _
2       one     one     NOUN    NN      _       0       root    _       _
3       .       .       PUNCT   .       _       2       punct   _       _
"""
    with open("dummy.conllu", "w", encoding="utf-8") as f:
        f.write(dummy_conllu_content)

    sentences = parse_conllu("dummy.conllu")
    for sent in sentences:
        print(sent)
        for token in sent.tokens:
            print(f"  {token}")
