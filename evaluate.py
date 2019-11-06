import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import CharacterTable, transform
from utils import restore_model, decode_sequences
from utils import read_text, tokenize

error_rate = 0.6
reverse = True
model_path = '/content/deep-spell-checkr/modelpl.h5'
hidden_size = 256
sample_mode = 'argmax'
data_path = '/content/deep-spell-checkr/data/'
books = ['ksiazki.txt.000', 'ksiazki.txt.001', 'ksiazki.txt.002']

test_sentence = 'Cetiozaurysk rodaj wymarlego dinozaura zauropoda zyjacego miedzy 166 a 164 milionow lat temu na terenach dzisiejszej Anglii. Bylo to czworonozna roslinozerca o umiarkowanym jak na zauropoda dlugim ogonie oraz dluzszych przednich lapach, dorownujacych tylnym. Jego dlugosc szacuje sie na 15 m, a mase na od 4 do 10 ton. Jedyna znna skamienialosc obejmuje wiekszosc tylnej polowy szkieletu wraz z zadnia konczyna. Znaziona zostala w Cambridgeshire w ostatniej dekadzie XIX wieku i opisana przez Arthura Smitha Woodwarda w 1905 roku jako nowy okaz gatunku cetiozaura Cetiosaurus leedsi. Jednak w 1927 rok Friedrich von Huene przeniosl to znalezisko do nowego rodzaju, ktoremu nadal nazwe Cetiosauriscus, co oznacza podobny do cetiozaura. Cetiosauriscus znaleziony zostala w osadah morskich formacji Oxford Clay wsrod szczatkow wielu roznych grup bezkregowcow, ichtiozaurow, plezjozaurow oraz krokodyli, a takze pojedynczego pterozaura i zroznicowanych dinozaurow'


if __name__ == '__main__':
    text  = read_text(data_path, books)
    vocab = tokenize(text)
    vocab = list(filter(None, set(vocab)))
    # `maxlen` is the length of the longest word in the vocabulary
    # plus two SOS and EOS characters.
    maxlen = max([len(token) for token in vocab]) + 2
    train_encoder, train_decoder, train_target = transform(
        vocab, maxlen, error_rate=error_rate, shuffle=False)

    tokens = tokenize(test_sentence)
    tokens = list(filter(None, tokens))
    nb_tokens = len(tokens)
    misspelled_tokens, _, target_tokens = transform(
        tokens, maxlen, error_rate=error_rate, shuffle=False)

    input_chars = set(' '.join(train_encoder))
    target_chars = set(' '.join(train_decoder))
    input_ctable = CharacterTable(input_chars)
    target_ctable = CharacterTable(target_chars)
    
    encoder_model, decoder_model = restore_model(model_path, hidden_size)
    
    input_tokens, target_tokens, decoded_tokens = decode_sequences(
        misspelled_tokens, target_tokens, input_ctable, target_ctable,
        maxlen, reverse, encoder_model, decoder_model, nb_tokens,
        sample_mode=sample_mode, random=False)
    
    print('-')
    print('Input sentence:  ', ' '.join([token for token in input_tokens]))
    print('-')
    print('Decoded sentence:', ' '.join([token for token in decoded_tokens]))
    print('-')
    print('Target sentence: ', ' '.join([token for token in target_tokens]))