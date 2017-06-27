# Standard library #


# ex 1
x = (2, 3, 4)
f= lambda x: min(x), max(x), sum(x)//len(x)
#print(f)

#print(min(2, 3, 4), max(2, 3, 4), sum(2, 3, 4)//len(2, 3, 4))

# ex 2

y = int(5)
num = y
num += 1
y == num
print(y)
if y == num:
	print("num is mutable")
else:
	print("num is immutable")

# int is immutable


y1 = str("word")
word = y1
word += "a"
y1 == word
print(y1)
if y1 == word:
	print("string is mutable")
else:
	print("string is immutable")
# str is immutable

y2 = [1, 2, 4]
list1 = y2
list1 == list1.append(1)
y2 == list1
print(y2)
if y2 == list1:
	print("list is mutable")
else:
	print("list is immutable")
# list is mutable

y3 = (1, 2, 4)
tuple1 = y3
tuple1 += (1,)
y3 == tuple1
print(y3)
if tuple1 == y3:
	print("tuple is mutable")
else:
	print("tuple is immutable")
# tuple is immutable


dict_1 = {1: 'x', 2: 'b'} # Create a dictionary.
dict_2 = dict_1 # Assign it a new name.
dict_2[1] = 'a' # Change the 'new' dictionary.
dict_1 == dict_2
print(dict_1)
if dict_1 == dict_2:
	print("dictionary is mutable")
else:
	print("dictionary is immutable")

# dictionary is mutable

# ex 3
import calculator as cl
import sys


a = 5
b = 12
c = cl.get_sqrt(cl.get_sum(cl.get_prod(a, a),cl.get_prod(b, b)))
print(c)


# ex 4
import random


name = None
while name == None:
    try:
    	name = str(input("Enter your name: "))
    except ValueError:
        print("This is not a valid name: ")
print("the palyer is : " + name)

numbers = list(range(1, 10)) # Get the integers from 1 to 9
print("numbers left: " + str(numbers))


# roll two dice

a = random.randint(1,6)
b = random.randint(1,6)
c = a + b
print("your number is: " + str(c))





