from pathlib import Path
from dataclasses import dataclass



print('Part 1: Dealing with the path library of Python')

my_path = Path('/home/siinvictus/projects/stupid_project/data/salary_data.xlsx')
print(f'The path is: {my_path}')
print(f'The parent folder is {my_path.parent}')
print(f'The actual file we are at is {my_path.name}')
print(f'The suffix of the file is {my_path.suffix}') #relevant to know the type of file you are working with or to check it if relevant.
print(f'At the moment, we are at {Path.cwd()} as a working directory.')

print('--------------------------------------------------------------------------------------------------------------------------')
print('Part 2: Dataclasses - used to create classes purely for data purposes that have the __repr__ and comparisons on their own ')

@dataclass
class Book:
    title: str
    author: str
    year: int

b1 = Book('Invisible Cities', 'Italo Calvino', 1972)
b2 = Book('Invisible Cities', 'Italo Calvino', 1972)
print(f'The first book is {b1}')
print(f'Are the two books, namely {b1} and {b2} the same? {b1==b2}')

@dataclass(frozen=True)
class Coordinates:
    latitude: str
    longitude: str

cord = Coordinates('45° 27\' 59" N', '9° 11\' 25" E' )
print(f'The coordinates for the center of Milan are {cord}')
#cord.longitude = 'something else'   #this will produce a dataclasses.FrozenInstanceError: cannot assign to field 'longitude' because you can't change once you've set it.
    

