# oop
# ex 1

class Backpack(object):
	def __init__(self, name, color, max_size=5.0):
		self.name = name 
		self.color = color
		self.max_size = max_size
		self.contents = []


	def put(self, item):
		if len(self.contents) > self.max_size:
			print('No Room!')
		else:
			self.contents.append(item)	



	def take(self, item):
		self.contents.remove(item)


	def dump(self):
		self.contents.remove()


	def test_backpack(name, color, max_size=5):
		testpack = Backpack("Barry", "black", 4) # Instantiate the object.
		if testpack.max_size != 5: # Test an attribute.
			print("Wrong default max_size!")
		for other in ("pencil", "pen", "paper", "computer"):
			testpack.put(item) # Test a method.
		print(testpack.contents)





# ex 2
class Jetpack(Backpack):
	def __init__(self, name, color, max_size=2, fuel=10):
		self.name = name
		self.color = color
		self.max_size = max_size
		self.fuel = fuel


	def fly(self, percent):
		if self.fuel - self.fuel * percent >0:
			self.fuel -= self.fuel * percent
		else:
			print("Not enough fuel!")

	
	def dump(self):
		self.contents.remove()


b = Backpack("Oscar", "green", 4)



#ex 3

	def __eq__(self, other):
	return len(self.contents) = len(other.contents)