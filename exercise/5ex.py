class library:
    def __init__(self):
        self.noBook = 0
        self.books = [] #empty list
    def addBook(self, book):
        self.books.append(book)
        self.noBook= len(self.books)
    def showInfo(self):
        print(f"the library has {self.noBook} books. tthe book are")
        for book in self.books:
            print(book)
l1 = library()
l1.addBook("python for Data Analysis")
l1.addBook("python Effected")
l1.addBook("python on hande book")
l1.showInfo()