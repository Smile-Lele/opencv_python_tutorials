import torch

with open('1342-0.txt', encoding='utf8') as f:
    text = f.read()

lines = text.split('\n')
line = lines[200]
print(line)

letter_tensor = torch.zeros(len(line), 128)
print(letter_tensor.shape)

for i, letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_tensor[i][letter_index] = 1


def clean_words(input_str):
    punctuation = '.,;:"!?“”_-'
    word_list = input_str.lower().replace('\n', ' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list


words_in_line = clean_words(line)
print(line)
print(words_in_line)

word_list = sorted(set(clean_words(text)))
word2index_dict = {word: i for i, word in enumerate(word_list)}
print(len(word2index_dict))
print(word2index_dict['michaelmas'])

word_tensor = torch.zeros(len(words_in_line), len(word2index_dict))
for i, word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_tensor[i][word_index] = 1
    print('{:2} {:4} {}'.format(i, word_index, word))

print(word_tensor.shape)
