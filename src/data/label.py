from enum import Enum, unique

@unique
class Label(Enum):
    AMBIENT = 0
    SYLLABLE = 1
    ECHEME = 2
    TRILL = 3
    CALL = 4
    NOISE = 5
    OTHER_SPECIES = 6

    @classmethod
    def valid(cls):
        return {cls.SYLLABLE.name, cls.ECHEME.name, cls.TRILL.name, cls.CALL.name, cls.NOISE.name}

    @classmethod
    def non_noise(cls):
        return {cls.SYLLABLE.name, cls.ECHEME.name, cls.TRILL.name, cls.CALL.name, cls.OTHER_SPECIES.name}
