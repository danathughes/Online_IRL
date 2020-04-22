

class DiscreteSpace:
	 """
	 A representation of a finite, discrete set of element.
	 """

	 def __init__(self, elements):
			"""
			Create a DiscreteSpace consisting of the provided elements
			
			elements - an iterable set of elements that define the space
			"""

			# Store the elements in a list, so elements can be enumerated
			self.elements = [element for element in elements]

			# Store a mapping from element to enumeration value
			self.element_map = { self.elements[i]:i for i in range(len(self.elements)) }


	 def size(self):
			"""
			Return the number of elements in the space.
			"""

			return len(self.elements)


	 def __len__(self):
			"""
			Return the number of elements in the space.  Equivalent to `size`
			"""

			return self.size()


	 def __iter__(self):
			"""
			Initialize iteration over the elements.  Allows the state space to be 
			iterated over.

			Example:
				 >>> space = DiscreteSpace(['a','b','c','d'])
				 >>> for element in space:
				 ...    print(element)
				 a
				 b
				 c
				 d
			"""

			# (Re)set an index over the elements
			self.__iter_idx = 0
			return self


	 def __next__(self):
			"""
			Get the next element.  See __iter__(self) docstring for details of method
			"""

			if self.__iter_idx < len(self.elements):
				 element = self.elements[self.__iter_idx]
				 self.__iter_idx += 1
				 return element
			else:
				 raise StopIteration


	 def __contains__(self, element):
			"""
			Returns a boolean whether the element is a member of the space.  Allows
			for using the `in` operator for querying element membership in the space.

			Example:
				 >>> space = DiscreteSpace(['a',b','c','d'])
				 >>> 'a' in space
				 True
				 >>> 'e' in space
				 False
			"""

			return element in self.elements


	 def __call__(self, element):
			"""
			Return the element number of the element

			element - element to find the index of.
			"""

			# TODO:  Create an assertion that the element is in the set of elements

			return self.element_map[element]


	 def __getitem__(self, index):
			"""
			Returns the element by index

			index - index of the desired element.
			"""

			# TODO: Make sure that the element exists

			return self.elements[index]



class DiscreteSpaceUnion:
	 """
	 A simple class used to create a union of discrete spaces, allowing simple
	 enumeration of a tuple of elements from each space.
	 """

	 def __init__(self, *args):
			"""
			Create an enumerator for the provided state spaces
			
			args - a sequence of DiscreteSpaces
			"""

			self.spaces = args


	 def __call__(self, *elements):
			"""
			"""

			# TODO: make sure the number of elements match the number of spaces

			return [space(element) for space,element in zip(self.spaces, elements)]


	 def __getitem__(self, indices):
			"""
			"""

			# TODO: make sure the number of indices match the number of spaces

			return [space[index] for space,index in zip(self.spaces, indices)]
