import enum

class Label(enum.Enum):
    AMBIENT = 0
    SYLLABLE = 1
    ECHEME = 2
    TRILL = 3
    CALL = 4
    NOISE = 5

    @classmethod
    def valid(cls):
        return  cls.SYLLABLE, cls.ECHEME, cls.TRILL, cls.CALL, cls.NOISE

    @classmethod
    def non_noise(cls):
        return cls.SYLLABLE, cls.ECHEME, cls.TRILL, cls.CALL
