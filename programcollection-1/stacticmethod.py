class Math:
    def __init__(self, num) -> None:
        self.num = num
    def multi(self, n) -> None:
        self.n = self.num * n
    @staticmethod
    def divi(a , b):
        return a / b
result = Math(2)
result.multi(2)
print(result.n)
print(Math.divi(2,98))