class Platesize:
    def __init__(self, rows, cols, sites):
        self.rowletters = rows
        self.colnumber = cols
        self.sitenumber = sites
sixwell = Platesize('ab', 3, 600)
twelvewell = Platesize('abc', 4, 300)
ninetysixwell = Platesize('abcdefgh', 12, 56)