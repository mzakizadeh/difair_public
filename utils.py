DATE_TOKEN_REGEX = r'\[DATE(\d+)(\+(\d+)(Y|M|D|h|m|s))?(:(Y|M|D|h|m|s))?\]'
NAME_TOKEN_REGEX = r'\[NAME(\d+)(:(F|M)?(1|L)?)?\]'

PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

def read_words_file(file_path):
    with open(file_path) as f:
        words = f.readlines()
    return list(map(lambda x: x.replace('\n', ''), words))

def words_to_tokens(words, tokenizer):
    return [tokenizer(w, add_special_tokens=False)['input_ids'][0] for w in words if len(tokenizer(w, add_special_tokens=False)['input_ids']) == 1]

def harmonic_mean(a, b):
    return 2 * b * a / (b + a)
