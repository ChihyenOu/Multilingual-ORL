class A():
    def __init__(self):
        self.first = B()

class B():
    def __init__(self):
        self.first_B = 11
        self.second_B = 22
        self.third_B = 33

if __name__ == "__main__":
    ex_A = A()
    for i, layer in enumerate(ex_A.first):
        print(i)
        print(layer)
