import copy
class Car:
    def __init__(self, a):
        self.a = a
    def add_student(self, b):
        self.b = b

jack = Car("lifan")
print("jack ", jack.a)
jork = copy.deepcopy(jack)
jork.a = "abay"
print("jack ", jack.a)
print("jork ", jork.a)