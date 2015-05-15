
import math

NOTE_SYMBOL = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_midi(freq: float):
    ''' Convert freqnecy in Hz to midi number
        A4 (440Hz) == 69
    '''
    return round(12*math.log2(freq/440) + 69)

def freq_to_note(freq: float) -> str:
    midi = freq_to_midi(freq)
    symbol = NOTE_SYMBOL[midi % 12]
    octave = int(midi/12 - 1)
    return symbol + str(octave)


if __name__ == "__main__":
    print(freq_to_note(440))
    print(freq_to_note(82))

